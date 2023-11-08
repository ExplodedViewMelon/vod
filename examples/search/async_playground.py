import asyncio


# Define an asynchronous process to be run repeatedly
async def repeated_task():
    while True:
        print("Running repeated task")
        await asyncio.sleep(2)  # Introduce a delay between repetitions


# Define an asynchronous process to monitor the condition
async def monitor_condition():
    await asyncio.sleep(10)  # Simulate waiting for another process to be done
    print("done awaiting")


async def main():
    # Create a task for the repeated process
    repeated_task_task = asyncio.create_task(repeated_task())

    # Create a task to monitor the condition
    monitor_task = asyncio.create_task(monitor_condition())

    await monitor_task

    # Cancel the repeated process task
    repeated_task_task.cancel()


# Run the event loop using asyncio.run()
if __name__ == "__main__":
    asyncio.run(main())
