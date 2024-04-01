#!/usr/bin/env python3

"""
Caveat when attempting to run the examples in non-gps environments:

`drone.offboard.stop()` will return a `COMMAND_DENIED` result because it
requires a mode switch to HOLD, something that is currently not supported in a
non-gps environment.
"""

import asyncio
import random
import time

from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)


async def run():
    MINUTES = 18000

    """ Does Offboard control using position NED coordinates. """

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()

    await asyncio.sleep(10)

    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed \
                with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    timeout = time.time() + 60*MINUTES #5 minutes
    while(time.time() < timeout):
        rand_yaw = random.random() * 360
        rand_n = random.random() * 10
        rand_e = random.random() * 10
        rand_d = random.random() * 5 + 10

        print(f"-- Go {rand_n}m North, {rand_e}m East, {-rand_d}m Down \
				within local coordinate system")
        print(f"Current time: {time.time()}, Target time: {timeout}")
        await drone.offboard.set_position_ned(
				PositionNedYaw(rand_n, rand_e, -rand_d, rand_yaw))
        await asyncio.sleep(5)

    '''print("-- Go 0m North, 10m East, 0m Down \
            within local coordinate system, turn to face South")
    await drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 10.0, 0.0, 180.0))
    await asyncio.sleep(10)'''

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed \
                with error code: {error._result.result}")
        
    print("-- landing")
    await drone.action.land()



if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())