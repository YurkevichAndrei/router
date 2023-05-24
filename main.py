import asyncio
import websockets
from geojson import LineString
from datetime import datetime
from route_module.main import Route

all_clients = []


# отправка сообщения клиенту
async def send_message(message, client_soket: websockets.WebSocketClientProtocol):
    await client_soket.send(message)


# расчет маршрута по точкам клиента с формированием geojson и его отправкой в ответ клиенту
async def route(message: str, client_soket: websockets.WebSocketClientProtocol):
    coord_mas = []
    coords = message.split("=>")
    for i in range(len(coords)-1):
        point = coords[i].split(":")
        point[0], point[1] = float(point[0]), float(point[1])
        coord_mas.append(point)
    coords_points = []
    h_points = []
    h_fly_points = []
    if len(coord_mas) != 0:
        start_time = datetime.now()
        print(start_time)
        route_with_h = Route(coord_mas[0][1], coord_mas[0][0], coord_mas[-1][1], coord_mas[-1][0]).get_route()
    # for i in range(len(coord_mas)-1):
    #     route_with_h = Route(coord_mas[i][1], coord_mas[i][0], coord_mas[i+1][1], coord_mas[i+1][0]).get_route()
    #     for j in range(len(route_with_h)):
    #         m_point = route_with_h[j]
    #         coords_points.append((m_point[1], m_point[0]))
        print(route_with_h)
        for m_point in route_with_h:
            coords_points.append((m_point[1], m_point[0]))
            h_points.append(m_point[2])
            h_fly_points.append(m_point[3])

        line = LineString(coords_points)
        line = str(line)[1:-1]

        geoj = "{" \
               "\"type\": \"FeatureCollection\"," \
               "\"name\": \"thing\"," \
               "\"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } }," \
               "\"features\": [" \
               "{ \"type\": \"Feature\", \"properties\": { }, \"geometry\": {" + line + "}" \
                                                                                        "}" \
                                                                                        "]" \
                                                                                        "}"
        print(geoj)
        print(datetime.now() - start_time)
        print('|=|'+str(h_points)+'|=|'+str(len(h_points))+'|=|'+str(h_fly_points))
        await send_message(geoj+'|=|'+str(h_points)+'|=|'+str(len(h_points))+'|=|'+str(h_fly_points), client_soket)


# обработка новых подуключений и новых сообщений
async def new_client_connected(client_soket: websockets.WebSocketClientProtocol, path: str):
    print("New connectetd!")
    all_clients.append(client_soket)
    while True:
        new_message = await client_soket.recv()
        print("New mess: ", new_message)
        if new_message == "client|:|close":
            client_soket.close()
        else:
            await route(new_message, client_soket)


# запуск сервера
async def start_server():
    await websockets.serve(new_client_connected, "localhost", 12345)

if __name__ == '__main__':
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(start_server())
    event_loop.run_forever()
