//
// Created by Charles Du on 11/10/22.
//

#include "Solver.h"

std::string get_stop_type_string(StopType t) {
    switch (t) {
        case Unknown: return "Unknown";
        case Xtol_Reached: return "Xtol_Reached";
        case Ftol_Reached: return "Ftol_Reached";
        case Gtol_Reached: return "Gtol_Reached";
        case Max_Iter_Reached: return "Max_Iter_Reached";
        case Custom_Criterion_Reached: return "Custom_Criterion_Reached";
        case Failure: return "Failure";
        case Success: return "Success";
        default: return "";
    }
}
