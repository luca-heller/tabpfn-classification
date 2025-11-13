# Tabpfn classification

TabPFN classification test case using the TabPFN model ([https://github.com/PriorLabs/TabPFN/tree/main](https://github.com/PriorLabs/TabPFN/tree/main)).


Run the test:

```bash
python tabpfn_classification.py \
  --csv taxi_data.csv \
  --output_dir results/ \
  --target payment_type \
  --features trip_distance ride_time fare_amount pickup_hour PULocationID DOLocationID passenger_count
```

Ref.

Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S., Schirrmeister, R., and Hutter, F. 2025. Accurate predictions on small data with a tabular foundation model. Nature.

Hollmann, N., Muller, S., Eggensperger, K., & Hutter, F. (2023). TabPFN: A transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations 2023.
