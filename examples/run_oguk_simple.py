"""
Simple OG-UK run script compatible with newer ogcore version.
This uses mostly default ogcore parameters with UK-specific tax calibration.
"""

import multiprocessing
from distributed import Client
import time
import os
import copy
import sys
import threading
from datetime import datetime
from oguk.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


class LiveTimer:
    """A live timer that updates in the terminal."""

    def __init__(self, message):
        self.message = message
        self.start_time = None
        self.running = False
        self.thread = None

    def _update_display(self):
        """Update the timer display."""
        while self.running:
            elapsed = time.time() - self.start_time
            # Use carriage return to overwrite the line
            sys.stdout.write(f"\r  [..] {self.message} [elapsed: {format_time(elapsed)}]   ")
            sys.stdout.flush()
            time.sleep(0.5)

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._update_display, daemon=True)
        self.thread.start()

    def stop(self, success_message=None):
        """Stop the timer and print final time."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        elapsed = time.time() - self.start_time
        # Clear the line and print success message
        sys.stdout.write("\r" + " " * 80 + "\r")  # Clear line
        if success_message:
            print(f"  [OK] {success_message} (completed in {format_time(elapsed)})")
        else:
            print(f"  [OK] {self.message} (completed in {format_time(elapsed)})")
        return elapsed


def run_with_timer(message, func, *args, **kwargs):
    """Run a function with a live timer."""
    timer = LiveTimer(message)
    timer.start()
    try:
        result = func(*args, **kwargs)
        timer.stop(message)
        return result
    except Exception as e:
        timer.stop(f"{message} - FAILED")
        raise e


def print_step(step_num, total_steps, description):
    """Print a formatted step header."""
    print("\n" + "=" * 70)
    print(f"  STEP {step_num}/{total_steps}: {description}")
    print("=" * 70)


def print_success(message, elapsed_time=None):
    """Print a success message."""
    if elapsed_time:
        print(f"  [OK] {message} (completed in {format_time(elapsed_time)})")
    else:
        print(f"  [OK] {message}")


def print_info(message):
    """Print an info message."""
    print(f"  [..] {message}")


def main(reform=None):
    total_start_time = time.time()
    total_steps = 6

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "           OG-UK MODEL SIMULATION".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # STEP 1: Initialize
    # =========================================================================
    print_step(1, total_steps, "INITIALIZATION")
    step_start = time.time()

    print_info("Starting Dask client...")
    client = Client(n_workers=1, threads_per_worker=1)
    num_workers = 1
    print_success(f"Dask client started with {num_workers} worker(s)")

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, "OG-UK-Simple", "OUTPUT_BASELINE")
    reform_dir = os.path.join(CUR_DIR, "OG-UK-Simple", "OUTPUT_REFORM")

    print_info("Creating output directories...")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(reform_dir, exist_ok=True)
    print_success(f"Baseline output: {base_dir}")
    print_success(f"Reform output: {reform_dir}")

    print_success("Initialization complete", time.time() - step_start)

    # =========================================================================
    # STEP 2: Set up baseline parameters
    # =========================================================================
    print_step(2, total_steps, "BASELINE PARAMETER SETUP")
    step_start = time.time()

    print_info("Creating Specifications object...")
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    print_success("Specifications object created")

    print_info("Setting model parameters...")
    print_info("  - tax_func_type: DEP")
    print_info("  - age_specific: False (single tax function for all ages)")
    print_info("  - start_year: 2026")
    print_info("  - J: 5 (quintiles)")

    p.update_specifications({
        "tax_func_type": "DEP",
        "age_specific": False,  # Single tax function (faster, compatible)
        "start_year": 2026,
        "J": 5,
    })
    print_success("Parameters updated")
    print_success("Baseline parameter setup complete", time.time() - step_start)

    # =========================================================================
    # STEP 3: Baseline tax function calibration
    # =========================================================================
    print_step(3, total_steps, "BASELINE TAX FUNCTION CALIBRATION")

    # Use live timer for the calibration
    timer = LiveTimer("Estimating baseline tax functions from PolicyEngine-UK")
    timer.start()

    c = Calibration(
        p, estimate_tax_functions=True, client=client
    )

    timer.stop("Baseline tax functions estimated")

    print_info("Updating specifications with tax parameters...")
    d = c.get_dict()
    updated_params = {
        "etr_params": d["etr_params"],
        "mtrx_params": d["mtrx_params"],
        "mtry_params": d["mtry_params"],
        "mean_income_data": d["mean_income_data"],
        "frac_tax_payroll": d["frac_tax_payroll"],
    }
    p.update_specifications(updated_params)
    print_success("Tax parameters applied to model")

    # =========================================================================
    # STEP 4: Run baseline steady state
    # =========================================================================
    print_step(4, total_steps, "BASELINE STEADY STATE COMPUTATION")

    # Use live timer for SS computation
    timer = LiveTimer("Computing baseline steady state (SS only, no TPI)")
    timer.start()

    runner(p, time_path=False, client=client)

    timer.stop("Baseline steady state computed")

    # =========================================================================
    # STEP 5: Reform calibration and steady state
    # =========================================================================
    print_step(5, total_steps, "REFORM POLICY SIMULATION")

    print_info("Setting up reform scenario...")
    p2 = copy.deepcopy(p)
    p2.baseline = False
    p2.output_base = reform_dir

    reform_dict = {
        "gov.hmrc.income_tax.rates.uk[0].rate": {
            "2023-01-01.2033-12-31": 0.30
        }
    }
    print_info("Reform: Income tax basic rate 20% -> 30%")

    # Use live timer for reform calibration
    timer = LiveTimer("Estimating reform tax functions")
    timer.start()

    c2 = Calibration(
        p2, iit_reform=reform_dict, estimate_tax_functions=True, client=client
    )

    timer.stop("Reform tax functions estimated")

    d2 = c2.get_dict()
    updated_params2 = {
        "etr_params": d2["etr_params"],
        "mtrx_params": d2["mtrx_params"],
        "mtry_params": d2["mtry_params"],
        "mean_income_data": d2["mean_income_data"],
        "frac_tax_payroll": d2["frac_tax_payroll"],
    }
    p2.update_specifications(updated_params2)
    print_success("Reform tax parameters applied")

    # Use live timer for reform SS
    timer = LiveTimer("Computing reform steady state")
    timer.start()

    runner(p2, time_path=False, client=client)

    timer.stop("Reform steady state computed")

    # =========================================================================
    # STEP 6: Results comparison
    # =========================================================================
    print_step(6, total_steps, "RESULTS COMPARISON")
    step_start = time.time()

    print_info("Loading results...")
    base_ss = safe_read_pickle(os.path.join(base_dir, "SS", "SS_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_ss = safe_read_pickle(os.path.join(reform_dir, "SS", "SS_vars.pkl"))
    reform_params = safe_read_pickle(os.path.join(reform_dir, "model_params.pkl"))
    print_success("Results loaded")

    # Print SS comparison results
    print("\n" + "=" * 70)
    print("  OG-UK STEADY STATE RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n  {'Variable':<15} {'Baseline':>15} {'Reform':>15} {'% Change':>15}")
    print("  " + "-" * 60)

    for var in ["Y", "C", "K", "L", "r", "w"]:
        if var in base_ss and var in reform_ss:
            base_val = float(base_ss[var])
            reform_val = float(reform_ss[var])
            pct_change = ((reform_val - base_val) / base_val) * 100
            print(f"  {var:<15} {base_val:>15.4f} {reform_val:>15.4f} {pct_change:>14.2f}%")

    print("  " + "-" * 60)
    print("\n  Reform: Income tax basic rate increased from 20% to 30%")
    print("=" * 70)

    print_success("Results comparison complete", time.time() - step_start)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_elapsed = time.time() - total_start_time

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "           SIMULATION COMPLETE".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {format_time(total_elapsed)}")
    print("\nOutput saved to:")
    print(f"  - Baseline: {base_dir}")
    print(f"  - Reform: {reform_dir}")
    print("#" * 70 + "\n")

    if client:
        client.close()

    return base_ss, reform_ss


if __name__ == "__main__":
    main()
