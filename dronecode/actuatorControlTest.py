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
import mavsdk
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw, PositionNedYaw, ActuatorControl, ActuatorControlGroup)
from torch import rand

async def run(numSteps):
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

    await drone.action.set_takeoff_altitude(7)
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

    for i in range(numSteps):
        print("Setting actuator control")
        #await drone.offboard.set_actuator_control(ActuatorControl([ActuatorControlGroup([1.0, 1.0, 1.0, 1.0])]))
        #await drone.offboard.set_actuator_control(ActuatorControl([ActuatorControlGroup([0,0,0,0,0,0,0,0]),ActuatorControlGroup([0,0,0,0,0,0,0,0])]))
        for i in range(1, 16):
            try:
                await drone.action.set_actuator(i, 1.0)
            except:
                print(f"Actuator index {i} didn't work")
        await asyncio.sleep(2)
    
    await asyncio.sleep(1.5)

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
    asyncio.run(run(20))