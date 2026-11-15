add_library(ramulator-controller OBJECT)

target_sources(
  ramulator-controller PRIVATE
  controller.h 
  scheduler.h 
  plugin.h
  refresh.h

  impl/dummy_controller.cpp
  impl/generic_dram_controller.cpp
  impl/pim_controller.cpp
  
  impl/scheduler/generic_scheduler.cpp
  impl/scheduler/pim_scheduler.cpp

  impl/refresh/all_bank_refresh.cpp
  impl/refresh/no_refresh.cpp

  impl/plugin/trace_recorder.cpp
  impl/plugin/cmd_counter.cpp

  )

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-controller
)
