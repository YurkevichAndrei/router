import math
import struct as st
from pathlib import Path
import multiprocessing as mp
import numpy as np
from osgeo import gdal
import networkx as nx
from datetime import datetime
from scipy.signal import savgol_filter
from scipy import interpolate


# класс хранения информации об изображении
class ImgData:
    def __init__(self, src_ds):
        self.width = src_ds.RasterXSize
        self.height = src_ds.RasterYSize
        gt = src_ds.GetGeoTransform()

        self.lon_min = gt[0]
        self.lat_min = gt[3] + self.width * gt[4] + self.height * gt[5]
        self.lon_max = gt[0] + self.width * gt[1] + self.height * gt[2]
        self.lat_max = gt[3]

        self.lat_step = abs(self.lat_max - self.lat_min) / self.height
        self.lon_step = abs(self.lon_max - self.lon_min) / self.width


class Route:
    def __init__(self, lat_start, lon_start, lat_finish, lon_finish):
        self.lat_start = lat_start
        self.lon_start = lon_start
        self.lat_finish = lat_finish
        self.lon_finish = lon_finish

        self.edge_len = 30  # метров
        self.diag_edge_len = self.edge_len * math.sqrt(2)
        self.k_horiz_v = 0.35
        self.k_vert_v = 1.0

        self.target_dir = Path(f"route_module\\files_h_{self.k_horiz_v}_v_{self.k_vert_v}\\")
        if not self.target_dir.exists():
            self.target_dir.mkdir(parents=True)

        self.target_filename = f"test_h_{self.k_horiz_v}_v_{self.k_horiz_v}"

    # возвращает матрицу изобрадения и данные о нем
    def get_height_matrix(self, name_image):
        src_ds = gdal.Open(name_image)
        img = src_ds.ReadAsArray()
        img = np.asmatrix(np.rollaxis(img, 0, 0))
        return img, ImgData(src_ds)

    # расчет веса ребра
    def get_edge_weight(self, dh, is_diag):
        # модуль изменения высоты * коэффициент вертикальной скорости +
        # + (длина деагонали если диагональ иначе длина стороны) * коэффициент горизонтальной скорости
        return abs(dh) * self.k_vert_v + (self.diag_edge_len if is_diag else self.edge_len) * self.k_horiz_v

    # номер пикселя по его расположению на изображении
    def get_node_number(self, row, column, width):
        return width * row + column

    # расположение пикселя на изображения по номеру пикселя
    def get_matrix_coords_by_number(self, num_pixel: int, width: int):
        column = num_pixel % width
        row = num_pixel // width
        return row, column

    # расчет и запись графа в файл
    def write_graph_to_files(self, h_matrix, row, h, w, filename):
        data_for_graph = []
        print(row)
        for column in range(w):
            cur_node_number = self.get_node_number(row, column, w)
            h_node = h_matrix[row - 1, column]
            # если не нулевая строка
            if row > 0:
                data_for_graph.append((cur_node_number, self.get_node_number(row-1, column, w),
                                       {'w': self.get_edge_weight(h_node - h_matrix[row-1, column], False)}))
                # если не последний столбец
                if column < w-1:
                    data_for_graph.append((cur_node_number, self.get_node_number(row - 1, column + 1, w),
                                           {'w': self.get_edge_weight(h_node - h_matrix[row-1, column+1], True)}))
                # если не нулевой столбец
                if column > 0:
                    data_for_graph.append((cur_node_number, self.get_node_number(row - 1, column - 1, w),
                                           {'w': self.get_edge_weight(h_node - h_matrix[row-1, column-1], True)}))
            # если не нулевой столбец
            if column > 0:
                data_for_graph.append((cur_node_number, self.get_node_number(row, column - 1, w),
                                       {'w': self.get_edge_weight(h_node - h_matrix[row, column - 1], False)}))
                # если не последняя строка
                if row < h-1:
                    data_for_graph.append((cur_node_number, self.get_node_number(row + 1, column - 1, w),
                                           {'w': self.get_edge_weight(h_node - h_matrix[row + 1, column - 1], True)}))
            # если не последняя строка
            if row < h-1:
                data_for_graph.append((cur_node_number, self.get_node_number(row + 1, column, w),
                                       {'w': self.get_edge_weight(h_node - h_matrix[row + 1, column], False)}))
                # если не последний столбец
                if column < w-1:
                    data_for_graph.append((cur_node_number, self.get_node_number(row + 1, column + 1, w),
                                           {'w': self.get_edge_weight(h_node - h_matrix[row + 1, column + 1], True)}))
            # если не последний столбец
            if column < w-1:
                data_for_graph.append((cur_node_number, self.get_node_number(row, column + 1, w),
                                       {'w': self.get_edge_weight(h_node - h_matrix[row, column + 1], False)}))

        f = (self.target_dir / f"{filename}_{row}.bin").open('wb')
        for data in data_for_graph:
            x, y = list(data[:-1])
            weight = data[-1]['w']
            try:
                f.write(x.to_bytes(4, 'big'))
                f.write(y.to_bytes(4, 'big'))
                f.write(st.pack('f', weight))
            except OSError:
                break
            except Exception as e:
                print(e)
                print(x, y, weight)
                break

        data_for_graph = []
        f.close()

    # перевод матрицы высот в граф с последующей записью в файлы
    def heights_to_graph(self, h_matrix: np.matrix, filename: str):
        print('heights_to_graph in')
        w = h_matrix.shape[1]
        h = h_matrix.shape[0]

        args = [(h_matrix, row, h, w, filename) for row in range(h)]
        with mp.Pool(mp.cpu_count()) as p:
            p.starmap(self.write_graph_to_files, args)

    # вывод бинарного файла с графом в консоль
    def print_bin(self, filename: str):
        f = Path(filename).open('rb')
        for i in range(15):
            data = f.read(12)
            p_1 = int.from_bytes(data[:4], byteorder='big')
            p_2 = int.from_bytes(data[4:8], byteorder='big')
            weight = st.unpack('f', data[8:])[0]
            print(p_1, p_2, weight, sep='\t')

    # получение интератора графа из бинарного файла
    def get_edges_from_bin(self, filename: Path):
        f = Path(filename).open('rb')
        data = f.read(12)
        while data:
            p_1 = int.from_bytes(data[:4], byteorder='big')
            p_2 = int.from_bytes(data[4:8], byteorder='big')
            weight = st.unpack('f', data[8:])[0]
            yield p_1, p_2, {'w': weight}
            data = f.read(12)
        return None

    # проверка расположения точки внутри изображения по географическим координатам
    def check_boundaries(self, lat: float, lon: float, img_data: ImgData):
        return img_data.lat_min < lat < img_data.lat_max and img_data.lon_min < lon < img_data.lon_max

    # перевод географтческих координат в матричные
    def geocoords_to_matrix_coords(self, lat: float, lon: float, img_data: ImgData):
        if not self.check_boundaries(lat, lon, img_data):
            raise Exception("точка вне снимка")
        column = int(abs(lon - img_data.lon_min) / img_data.lon_step)
        row = int(abs(img_data.lat_max - lat) / img_data.lat_step)
        return row, column

    # перевод матричных координат в географические
    def matrix_coords_to_geocoords(self, row: int, column: int, img_data: ImgData):
        lat = img_data.lat_max - row * img_data.lat_step
        lon = img_data.lon_min + column * img_data.lon_step
        return lat, lon

    # проверка расположения точки внутри изображения по матричным координатам
    def check_matrix_boundaries(self, row, column, img_data: ImgData):
        return 0 <= row < img_data.height, 0 <= column < img_data.width

    # получение графа на площадь для построения маршрута
    def get_area_edges(self, img_data: ImgData, row_start: int, column_start: int, row_finish: int, column_finish: int):
        indent = 10

        # проверяем, что точка старта расположена ближе к точке (0, 0)
        # если нет,меняе местами координаты точки старта и финиша
        if row_start > row_finish:
            row_start, row_finish = row_finish, row_start
        if column_start > column_finish:
            column_start, column_finish = column_finish, column_start

        min_row = row_start - indent
        min_column = column_start - indent
        min_result = self.check_matrix_boundaries(min_row, min_column, img_data)
        if not min_result[0]:
            min_row = 0
        if not min_result[1]:
            min_column = 0

        max_row = row_finish + indent
        max_column = column_finish + indent
        max_result = self.check_matrix_boundaries(max_row, max_column, img_data)
        if not max_result[0]:
            max_row = img_data.height - 1
        if not max_result[1]:
            max_column = img_data.width - 1

        args = [(num_line, min_row, max_row, min_column, max_column, img_data.width) for num_line in range(min_row, max_row + 1)]
        with mp.Pool(mp.cpu_count()) as p:
            return p.starmap(self.get_edges_from_line, args), max_row - min_row, max_column - min_column

    # получения графа одной строки матрицы на площадь построения маршрута
    def get_edges_from_line(self, num_line: int, min_row: int, max_row: int, min_column: int, max_column: int, widht: int):
        file = self.target_dir / f'{self.target_filename}_{num_line}.bin'
        result = []
        for edge in self.get_edges_from_bin(file):
            p_1, p_2, _ = edge
            _, p_1_column = self.get_matrix_coords_by_number(p_1, widht)
            p_2_row, p_2_column = self.get_matrix_coords_by_number(p_2, widht)

            if (min_column <= p_1_column <= max_column) and \
               (min_column <= p_2_column <= max_column) and \
               (min_row <= p_2_row <= max_row):
                result.append(edge)
        return result

    # получение точек старта и финиша
    def get_points(self):
        return self.lat_start, self.lon_start, self.lat_finish, self.lon_finish

    def validate_h_list(self, h_list: list, h_list_smooth: list, not_valid_handler):
        for i in range(len(h_list)):
            if h_list[i] > h_list_smooth[i]:
                not_valid_handler(i)
                return False
        return True

    def smoothing_coords(self, route_list: list, i: int, matrix, img_data: ImgData):
        lat_0, lon_0 = route_list[i - 1]
        lat_1, lon_1 = route_list[i]
        lat_2, lon_2 = route_list[i + 1]

        lat_1 = (lat_0 + lat_1 + lat_2) / 3
        lon_1 = (lon_0 + lon_1 + lon_2) / 3

        row, column = self.geocoords_to_matrix_coords(lat_1, lon_1, img_data)
        height = matrix[row, column]
        return [lat_1, lon_1, height]

    def moving_average(self, l: list, window_size: int):
        result = []
        if window_size == 0:
            return l

        for ind in range(len(l)):
            sum = 0
            for m_ind in range(ind - (window_size // 2), ind + (window_size // 2) + (window_size % 2)):
                if m_ind < 0:
                    sum += l[0]
                elif m_ind >= len(l):
                    sum += l[-1]
                else:
                    sum += l[m_ind]
            result.append(sum / window_size)
        return result

    def smooth_route(self, route_list: list, matrix, img_data):
        print('smoothing')
        result_smooth = []
        lat, lon = route_list[0]
        # matrix, img_data = self.get_height_matrix('route_module/image.tif')
        row, column = self.geocoords_to_matrix_coords(lat, lon, img_data)
        height = matrix[row, column]
        result_smooth.append([lat, lon, height])

        result_smooth = result_smooth + [self.smoothing_coords(route_list, i, matrix, img_data) for i in range(1, len(route_list) - 1)]

        lat, lon = route_list[-1]
        row, column = self.geocoords_to_matrix_coords(lat, lon, img_data)
        height = matrix[row, column]
        result_smooth.append([lat, lon, height])

        result_smooth[0].append(result_smooth[0][2]+10)

        h_list_origin = [r[2]+10 for r in result_smooth]
        h_list = h_list_origin[:]
        # num_h = [i for i in range(len(h_list))]
        # h_list_sm = savgol_filter(h_list, len(h_list) // 5, 3)
        h_list_sm_first = self.moving_average(h_list, len(h_list) // 3)
        h_list_sm = h_list_sm_first.copy()

        def not_valid_handler(index: int):
            h_list[index] += 10

        while not self.validate_h_list(h_list_origin, list(h_list_sm), not_valid_handler):
            print("not valid")
                    # h_list[i] = result_smooth[i][2]+10
            # h_list_sm = savgol_filter(h_list, len(h_list) // 5, 3)
            h_list_sm = self.moving_average(h_list, len(h_list) // 3)

        h_list_sm_first = h_list_sm.copy()
        #     print("not valid")
        #     for i in range(len(h_list)):
        #         if h_list_origin[i] > h_list_sm[i]:
        #             h_list_sm[i] += 10

        def not_valid_handler_sm(index: int):
            h_list_sm[index] += 10

        h_list_sm_sm = self.moving_average(h_list_sm, len(h_list) // 5)

        while not self.validate_h_list(h_list_origin, list(h_list_sm_sm), not_valid_handler_sm):
            print("not valid sm")
            h_list_sm_sm = self.moving_average(h_list_sm, len(h_list) // 5)

        # tck = interpolate.splrep(num_h, h_list, k=2, s=500)
        # h_list = interpolate.splev(num_h, tck, der=0)
        for i in range(1, len(result_smooth) - 1):
            result_smooth[i].append(h_list_sm_sm[i])

        # for i in range(1, len(result_smooth)-1):
        #     h1 = result_smooth[i][2]
        #     h2 = result_smooth[i+1][2]
        #     h3 = 0
        #     h4 = 0
        #     h5 = 0
        #     dh2 = abs(h2 - h1)
        #     dh3 = 0
        #     dh4 = 0
        #     dh5 = 0
        #     if len(result_smooth) - i - 4 >= 0:
        #         h3 = result_smooth[i+3][2]
        #         dh3 = abs(h3 - h1)
        #     if len(result_smooth) - i - 5 >= 0:
        #         h4 = result_smooth[i+4][2]
        #         dh4 = abs(h4 - h1)
        #     if len(result_smooth) - i - 6 >= 0:
        #         h5 = result_smooth[i+5][2]
        #         dh5 = abs(h5 - h1)
        #     dh = max(dh2, dh3, dh4, dh5)
        #     if dh != 0:
        #         sum_h = 0.0
        #         kh = math.ceil(dh / 10)
        #         count_p = math.ceil(kh/2)
        #         # проверка границ маршрута
        #         if count_p > i+1:
        #             count_p = i+1
        #             kh = count_p * 2
        #         elif count_p > len(result_smooth) - (i + 1):
        #             count_p = (len(result_smooth) - (i + 1))
        #             kh = count_p * 2
        #         print(f'h1 {h1} h2 {h2} h3 {h3} h4 {h4} h5 {h5} dh {dh} kh {kh} coun_p {count_p}')
        #
        #         l_max_i = i
        #         l_max_is_found = False
        #         r_max_i = i
        #         r_max_is_found = False
        #         for k in range(count_p):
        #             if not l_max_is_found:
        #                 if result_smooth[i - k - 1][2] <= result_smooth[i - k][2] > result_smooth[i - k + 1][2]:
        #                     l_max_i = i - k
        #                     l_max_is_found = True
        #             if not r_max_is_found:
        #                 if result_smooth[i + k - 1][2] < result_smooth[i + k][2] >= result_smooth[i + k + 1][2]:
        #                     l_max_i = i + k
        #                     l_max_is_found = True
        #
        #         if not l_max_is_found:
        #             l_max_i = i - count_p
        #
        #         if not r_max_is_found:
        #             r_max_i = i + count_p
        #
        #         print(f"{l_max_i=}, {r_max_i=}")
        #
        #         c_p = 0
        #         for j in range(l_max_i, r_max_i+1):
        #             sum_h += result_smooth[j][2]
        #             print(j)
        #             c_p += 1
        #         print(f'{h1}  {sum_h}/{c_p} + 10 = {int(sum_h)/c_p + 10}')
        #         result_smooth[i].append((int(sum_h) / c_p) + 10)
        #
        #     else:
        #         result_smooth[i].append(h1 + 10)

        result_smooth[-1].append(result_smooth[-1][2] + 10)

        print('smoothing end')
        return h_list_sm_first, result_smooth

    # получение маршрута
    def get_route(self):
        matrix, m_data = self.get_height_matrix('route_module/image.tif')
        start_coord = self.geocoords_to_matrix_coords(self.lat_start, self.lon_start, m_data)
        finish_coord = self.geocoords_to_matrix_coords(self.lat_finish, self.lon_finish, m_data)
        edges, height, widht = self.get_area_edges(m_data, *start_coord, *finish_coord)
        res_edges = []
        for l in edges:
            res_edges.extend(l)

        graph = nx.Graph()
        # добавление номеров вершин в граф
        for n1, n2, _ in res_edges:
            graph.add_node(n1)
            graph.add_node(n2)
        # добавление ребер в граф
        graph.add_edges_from(res_edges)

        # получение точек маршрута
        points_route = nx.shortest_path(graph, source=self.get_node_number(*start_coord, m_data.width),
                                        target=self.get_node_number(*finish_coord, m_data.width), weight="w")
        # перевод точек маршрута в географические координаты и считываение высоты в них
        result = []
        for i, point in enumerate(points_route):
            row, column = self.get_matrix_coords_by_number(point, m_data.width)
            lat, lon = self.matrix_coords_to_geocoords(row, column, m_data)
            if i == 0:
                lat, lon = self.lat_start, self.lon_start
                print("start", lat, lon)
            elif i == len(points_route) - 1:
                lat, lon = self.lat_finish, self.lon_finish
            result.append((lat, lon))
        return self.smooth_route(result, matrix, m_data)


if __name__ == '__main__':
    start_time = datetime.now()
    route = Route(55.63991, 37.50287, 55.57371, 37.59204)
    mat, _ = route.get_height_matrix('image.tif')
    route.heights_to_graph(mat, route.target_filename)
    print(datetime.now() - start_time)
