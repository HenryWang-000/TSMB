import enums
import generate_results_function


# ============  Merge Results  ============
dataset_name1 = enums.DataSetName.OCCUPANCY.value
dataset_name2 = enums.DataSetName.PUMP_SENSOR.value
dataset_name3 = enums.DataSetName.POWER_DEMAND.value
dataset_name4 = enums.DataSetName.AIR_QUALITY.value



tsmb_top, perturbed_top, tdb_top, \
tsmb_select_b, perturbed_select_b, tdb_select_b, \
confidence_tsmb, all_performance, performance, \
tft_performance, tft_select_b = generate_results_function.merge_results(dataset_name1, dataset_name2, dataset_name3, dataset_name4)


tsmb_top.to_csv('tsmb_top.csv', index=False)
perturbed_top.to_csv('perturbed_top.csv', index=False)
tdb_top.to_csv('tdb_top.csv', index=False)
tsmb_select_b.to_csv('tsmb_select_b.csv', index=False)
perturbed_select_b.to_csv('perturbed_select_b.csv', index=False)
tdb_select_b.to_csv('tdb_select_b.csv', index=False)
confidence_tsmb.to_csv('confidence_tsmb.csv', index=False)
all_performance.to_csv('all_performance.csv', index=False)
performance.to_csv('performance.csv', index=False)
tft_performance.to_csv('tft_performance.csv', index=False)
tft_select_b.to_csv('tft_select_b.csv', index=False)