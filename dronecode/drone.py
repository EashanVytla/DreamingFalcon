from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw, PositionNedYaw, ActuatorControl)
import asyncio

class Drone:
    def __init__(self, drone, replay):
        self.drone = System()
        self.replay = replay
        
    async def set_powers(self, a1, a2, a3, a4):
        await self.drone.offboard.set_actuator_control(ActuatorControl({1.0, 1.0, 1.0, 1.0}))
        self.replay.add()
    
    async def init(self):
        """ Does Offboard control using position NED coordinates. """
        drone = System()
        await self.drone.connect(system_address="udp://:14540")

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
        await self.drone.action.arm()

        print("-- Taking off")
        await self.drone.action.takeoff()

        await asyncio.sleep(10)

        print("-- Setting initial setpoint")
        await self.drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

        print("-- Starting offboard")
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard mode failed \
                    with error code: {error._result.result}")
            print("-- Disarming")
            await self.drone.action.disarm()
            return