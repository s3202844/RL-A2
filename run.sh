# # baseline
# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.1 -g=0.8 -s=egreedy -e=0.2 -c=1.0`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# # learning rate
# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.2 -g=0.8 -s=egreedy -e=0.2 -c=1.0`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.4 -g=0.8 -s=egreedy -e=0.2 -c=1.0`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# # gamma
# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.1 -g=1 -s=egreedy -e=0.2 -c=1.0`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# # epsilon
# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.1 -g=0.8 -s=egreedy -e=0.05 -c=1.0`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# python dqn.py -l=0.1 -g=0.8 -s=egreedy -e=0.4 -c=1.0
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# # tempture
# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.1 -g=0.8 -s=softmax -e=0.2 -c=0.1`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.1 -g=0.8 -s=softmax -e=0.2 -c=1.0`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# curr=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start Time: $curr"
# start_time=$(date +%s)
# `python dqn.py -l=0.1 -g=0.8 -s=softmax -e=0.2 -c=10`
# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

# additional
curr=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start Time: $curr"
start_time=$(date +%s)
python dqn.py -r
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

curr=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start Time: $curr"
start_time=$(date +%s)
python dqn.py -t
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"

curr=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start Time: $curr"
start_time=$(date +%s)
python dqn.py -r -t
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Spent Time: $(($cost_time/60))min $(($cost_time%60))s"