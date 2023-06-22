import asyncio
import websockets
from geojson import LineString
from datetime import datetime
from route_module.main import Route
import simplekml as skml
from pathlib import Path

all_clients = []


# отправка сообщения клиенту
async def send_message(message, client_soket: websockets.WebSocketClientProtocol):
    await client_soket.send(message)


async def send_kml(client_soket: websockets.WebSocketClientProtocol):
    target_dir = Path(f"files\\")
    f = target_dir / f"{client_soket.id}.kml"
    with open(f, "r") as read_file:
        data = read_file.read()
    await client_soket.send(f"kml|=|{data}")


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
    points_for_file = []
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

        target_dir = Path(f"files\\")
        if not target_dir.exists():
            target_dir.mkdir(parents=True)

        for m_point in route_with_h:
            coords_points.append((m_point[1], m_point[0]))
            h_points.append(m_point[2])
            h_fly_points.append(m_point[3])
            points_for_file.append((m_point[1], m_point[0], m_point[3] - route_with_h[0][2]))
            # points_for_file.append((m_point[1], m_point[0], m_point[3] - m_point[2]))

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
        f = target_dir / f"{client_soket.id}.kml"
        kml = skml.Kml()
        linestring = kml.newlinestring(name=f"Route_{client_soket.id}")
        linestring.coords = points_for_file
        # linestring.altitudemode = skml.AltitudeMode.clamptoground
        linestring.altitudemode = skml.AltitudeMode.relativetoground
        linestring.extrude = 1
        kml.save(f)
        await send_message('Route|=|' + geoj+'|=|'+str(h_points)+'|=|'+str(len(h_points))+'|=|'+str(h_fly_points), client_soket)


# обработка новых подуключений и новых сообщений
async def new_client_connected(client_soket: websockets.WebSocketClientProtocol, path: str):
    print("New connectetd!")
    all_clients.append(client_soket)
    while True:
        new_message = await client_soket.recv()
        print("New mess: ", new_message)
        if new_message == "client|:|close":
            client_soket.close()
        elif new_message == "create|:|kml":
            await send_kml(client_soket)
        else:
            await route(new_message, client_soket)


# запуск сервера
async def start_server():
    await websockets.serve(new_client_connected, "localhost", 12345)

if __name__ == '__main__':
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(start_server())
    event_loop.run_forever()
