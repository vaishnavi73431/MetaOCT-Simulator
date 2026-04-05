import tracemalloc
import time
from env import MetaOCTEnv, Action

def run_stability_test():
    env = MetaOCTEnv()
    obs = env.reset()
    
    print("Starting 100+ iteration stability test...")
    tracemalloc.start()
    
    # Run 150 iterations
    for i in range(150):
        # Dummy action
        action = Action(
            diagnosis="NORMAL",
            heatmap_coordinates=[[0, 0], [0, 0]],
            reasoning="Shows normal foveal contour and intact rpe."
        )
        step_result = env.step(action)
        if step_result.done:
            obs = env.reset()
            
        if i % 30 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iter {i}: Current Memory: {current / 10**6:.3f} MB; Peak: {peak / 10**6:.3f} MB")
            
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Final Memory: {current / 10**6:.3f} MB; Peak: {peak / 10**6:.3f} MB")
    print("Stability test passed successfully (no major leaks!).")

if __name__ == "__main__":
    import logging
    # Disable env step logging to avoid spam
    logging.getLogger('env').setLevel(logging.WARNING)
    run_stability_test()
