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
import sys

from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw, PositionNedYaw)
from torch import rand


async def runMixed(numSteps):
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

    for i in range(numSteps):
        rand_yaw = random.random() * 5
        rand_n = random.random() * 10
        rand_e = random.random() * 10
        rand_d = random.random() * 5 + 15

        print(f"-- Go {rand_n}m North, {rand_e}m East, {-rand_d}m Down \
				within local coordinate system")
        print(f"Current time: {i}, Target steps: {numSteps}")
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

async def runInd(numSteps):
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

    await drone.action.set_takeoff_altitude(24)
    await drone.action.set_maximum_speed(20)

    print("-- Taking off")
    await drone.action.takeoff()

    await asyncio.sleep(15)

    max_speed = await drone.action.get_maximum_speed()

    print(f"Max Speed: {max_speed}")

    print("-- Setting initial setpoint")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed \
                with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    for i in range(numSteps[0]):
        rand_yaw = random.random() * 360

        print(f"-- Go {rand_yaw}degrees Yaw")
        print(f"Current step: {i}, Target steps: {numSteps[0]}")
        await drone.offboard.set_velocity_ned(
				VelocityNedYaw(0, 0, 0, rand_yaw))
        await asyncio.sleep(2)

    await drone.offboard.set_velocity_ned(
				VelocityNedYaw(0, 0, 0, 0))
    
    await asyncio.sleep(1.5)

    rand = [0.0, 0.0, 0.0]

    for i in range(3):
        for j in range(numSteps[i]):
            if(j % 2 == 0):
                rand[i] = random.uniform(2, max_speed)
            else:
                rand[i] = -random.uniform(2, max_speed)

            print(f"-- Go {rand[0]}m/s north, {rand[1]}m/s east, {rand[2]}m/s down")
            print(f"Current step: {j}, Target steps: {numSteps[i]}")
            await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(rand[0], rand[1], rand[2], 0.0))
            await asyncio.sleep(1.5)
            await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(2)
        rand[i] = 0.0

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
    if len(sys.argv) < 2:
        print("Usage: python script.py <num minutes> <motion type i(independent) or m(mixed)>")
        sys.exit(1)

    if(sys.argv[1] == 'm'):
        asyncio.run(runMixed(400))
    else:
        asyncio.run(runInd([80, 80, 80, 80]))