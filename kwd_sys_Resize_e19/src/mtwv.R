#!/usr/bin/env Rscript

# Usage:
#
#    Rscript src/mtwv.R exp_results.csv 71839 > exp_atwv.csv
#
# Note the 71839 is the argument for refs_seconds, which is the length of the search corpus in seconds
# Additional parameters to the mtwv function can be supplied as well, e.g. (cost_false_alarm = 1, cost_missed_detection = 10):
#
#    Rscript src/mtwv.R exp_results.csv 71839 1 10 > exp_atwv.csv
#
# See function definition below for full argument list

if(!"dplyr" %in% installed.packages()) { install.packages("dplyr")  }

suppressMessages(library(dplyr))

mtwv <- function(exp_results, refs_seconds, cost_false_alarm = 1, cost_missed_detection = 100, th_start = 0.7, th_end = 0.95, th_step = 0.05) {
  
  seq(th_start, th_end, th_step) %>%
    purrr::map_dfr(function(threshold) {
      
      true_positive_prior <- sum(exp_results$label)/length(exp_results$label)
      beta <- (cost_false_alarm / cost_missed_detection) * ((1/true_positive_prior) - 1)
      
      exp_results %>%
        # Get total number of true positives for each query
        dplyr::group_by(query) %>%
        dplyr::arrange(query, desc(pred)) %>% 
        dplyr::mutate(
          tpos_count = sum(label),
          retrieved  = pred >= threshold,
          correct    = (label == 1) & retrieved,
          spurious   = retrieved & (label == 0)
        ) %>% 
        dplyr::group_by(query) %>% 
        dplyr::summarise(
          n_retrieved   = sum(retrieved),
          n_correct     = sum(correct),
          tpos_count    = unique(tpos_count),
          precision     = ifelse(n_retrieved != 0, n_correct / n_retrieved, NA),
          recall        = n_correct / tpos_count,
          p_miss        = 1 - (n_correct / tpos_count),
          n_nontarget   = refs_seconds - tpos_count,
          n_spurious    = sum(spurious),
          p_false_alarm = (n_spurious / n_nontarget),
          twv_q         = 1 - p_miss - (beta * p_false_alarm),
          .groups = "drop"
        ) %>% 
        dplyr::ungroup() %>% 
        # system level
        dplyr::summarise(
          theta      = threshold,
          actual_twv = round(mean(twv_q, na.rm = TRUE), 3),
          avg_prec   = mean(precision, na.rm = TRUE),
          avg_rec    = mean(recall, na.rm = TRUE),
          .groups = "drop"
        ) 
    }) %>%
    dplyr::arrange(desc(actual_twv))
  
}

# Check if being called as a script
args <- commandArgs()

if(any(stringr::str_detect(args, "mtwv.R"))) {
  
  trail_args  <- commandArgs(trailingOnly = TRUE)
  
  # Only first argument, the CSV file path is a string
  # The rest are numbers (integers/double)
  func_params <- as.double(trail_args[-1])
  
  exp_results <- readr::read_csv(trail_args[1], col_types = readr::cols())
  
  if("epoch" %in% names(exp_results)) {
    
    exp_results <- split(exp_results, exp_results$epoch)
    
    return_df <- purrr::imap_dfr(
      .x = exp_results,
      .f = function(epoch_results, epoch_no) {
        
        dplyr::bind_cols(
          tibble::tibble(epoch = as.integer(epoch_no)),
          do.call("mtwv", list(epoch_results, func_params))
        )
        
      }
    )
    
  } else {
    
    return_df <- do.call("mtwv", list(exp_results, func_params))
    
  }
  
  writeLines(readr::format_csv(return_df), stdout())
  
} else {
  # Remove args object if sourcing function from RStudio
  rm(args)
}