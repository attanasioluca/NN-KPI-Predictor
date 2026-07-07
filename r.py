import os, json, copy, time
import numpy as np
from helpers.simulator import ScenarioSimulator
import logging

# ---------------------------------------------------------
# FILE-BASED LOGGING: Forces output to a file so it cannot 
# be swallowed by terminal buffering or process crashes.
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debug_simulation.log", mode='w'),
        logging.StreamHandler()
    ]
)

def run_single_diagnostic():
    logging.info("--- STARTING SINGLE-THREADED DIAGNOSTIC RUN ---")
    BASE_FILE = "data/real/model/scenario.json"
    MODEL_FILE = "data/real/model/model.json"

    # Step 1: File Loading
    logging.info("Step 1: Loading JSON files...")
    try:
        with open(BASE_FILE, 'r') as f:
            base_json = json.load(f)
        with open(MODEL_FILE, 'r') as f:
            full_model = json.load(f)
        logging.info("-> Files loaded successfully.")
    except Exception as e:
        logging.error(f"-> FAILED to load files: {e}")
        return

    # Step 2: Extracting details
    logging.info("Step 2: Parsing process details...")
    try:
        process_details = {}
        for p_id, p_data in full_model.get("process_elements", {}).items():
            if p_data.get("node_details"):
                process_details = p_data
                break

        valid_node_ids = set(process_details["node_details"].keys())
        for node_id, node in process_details["node_details"].items():
            if "previous" in node and node["previous"]:
                node["previous"] = [p for p in node["previous"] if p in valid_node_ids]
        logging.info("-> Process details parsed successfully.")
    except Exception as e:
        logging.error(f"-> FAILED to parse process details: {e}")
        return

    # Step 3: Base Setup
    logging.info("Step 3: Setting up the baseline configuration...")
    scenario_id = 999  # Test ID
    variant = copy.deepcopy(base_json)
    base = variant["0"]

    # We will force a SAFE arrival rate to rule out the million-instance explosion
    arr_dist = base.get("arrivalRateDistribution", {})
    if arr_dist and "mean" in arr_dist:
        base["arrivalRateDistribution"]["mean"] = "600"
    
    logging.info("-> Configuration set with safe arrival rate (600s).")

    # Step 4: Simulator Initialization
    logging.info("Step 4: Instantiating ScenarioSimulator...")
    try:
        simulator = ScenarioSimulator(base, full_model, process_details, seed=scenario_id)
        logging.info("-> Simulator instantiated successfully.")
    except Exception as e:
        logging.error(f"-> FAILED to initialize Simulator: {e}")
        return

    # Step 5: Execution
    logging.info("Step 5: Firing run_scenario() for 1 replication...")
    try:
        start_time = time.time()
        # Limiting to 30 simulated days
        result = simulator.run_scenario(replications=1, until=86400 * 30)
        elapsed = time.time() - start_time
        
        logging.info(f"--- SUCCESS ---")
        logging.info(f"Simulation completed perfectly in {elapsed:.2f} real seconds.")
        logging.info(f"Outputs: {result}")
        
    except Exception as e:
        logging.error(f"-> SIMULATOR CRASHED DURING EXECUTION: {e}")

if __name__ == "__main__":
    run_single_diagnostic()