# Data Information
save train data files according file types
## Folder List
* ### external
  * save external results to .csv
  * e.g. external_results_{workload number}.csv
  ```
  external_results_2.csv
  ```
  * Also, save default external results on according workload
  ```
  default_external.csv
  ```
* ### internal
  * save internal results to .csv
  * e.g. internal_results_{workload number}.csv
  ```
  internal_results_2.csv
  ```
* ### rocskdb_conf
  * save knobs configuration to .cnf, we used 20,000 samples
  * e.g. config{number}.cnf
  ```
  config1.cnf
  config20000.cnf
  ```
* ### target_workload
  * save target workload information named number folder
  * e.g. target_workload/{target workload number}/{results}.csv
   ```
   target_workload/16/external_results_11.csv
   target_workload/16/internal_results_11.csv
   ```
* ### lookuptable
  * save gained lookuptable from training onehot vector to internal metrics in similar workload
  * e.g. lookuptable/{similar workload number}/LookupTable.npy
  ```
  lookuptable/1/LookupTable.npy
  ```
* ### KnobsOneHot.npy
  * save pre-gained one-hot vector information of knobs to reduce run-time
