fname="results.csv"
echo "approach,query_type,sparsity,size,time" > $fname
for query_type in "rank" "select"; do
    for sparsity in 10 50 90; do
        for size in 24 26 28 30 32 34; do
            full_size=$((2 ** $size))
            for run in {1..5}; do
                ./bin/orzo-benchmark $query_type $full_size $sparsity $run >> $fname
            done
        done
    done
done
