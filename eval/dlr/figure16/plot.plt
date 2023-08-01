#!/usr/bin/env gnuplot

dat_file="data.dat"
set datafile sep '\t'
set output "data.eps"

col_app                          = 1
col_cache_policy                 = 2
col_cache_percent                = 3
# col_batch_size                   = 4
col_dataset                      = 5
col_seq                          = 6
col_seq_time_feat_copy           = 7
col_seq_time_train_total         = 8
col_seq_time_duration            = 9
col_refresh_bucket_sz            = 10
col_enable_refresh               = 11
set terminal postscript "Helvetica,16" eps enhance color dl 2 background "white"

set pointsize 1
# set size 0.4,0.35
set size 0.4,0.35
# set size 2.4,1.05

set tics font ",14" scale 0.5

format_str="<awk -F'\\t' '{ if(". \
                              "$".col_enable_refresh."      ~ \"%s\"     && ". \
                              "$".col_refresh_bucket_sz."   ~ \"%s\"     ". \
                              ") { print }}' %s "
cmd_filter_dat(file, refresh, bucket_sz)=sprintf(format_str, refresh, bucket_sz, file)


# ### Key
set key inside left Left top enhanced nobox reverse # autotitles columnhead
set label 
set key samplen 1.5 spacing 1.5 height 0.2 width -2 font ',13' maxrows 4 center at graph 0.7, graph 0.7 #maxrows 1 at graph 0.02, graph 0.975  noopaque
# unset key

## Y-axis
set ylabel "Inference Time(ms)" offset 2.5,0
set yrange [0:]
# set ytics 0,1,5 
set ytics offset 0.5,0 #format "%.1f" #nomirror
set grid y

### X-axis
set xlabel "Timeline(s)" offset 0,0.7
set xtics nomirror offset 0,0.3

bucket_sz="8000"

set arrow from first refresh_start, graph 0 to first refresh_start, graph 1 nohead lt 1 lw 3 dashtype(3,2) lc "#000000"
set arrow from first refresh_stop,  graph 0 to first refresh_stop,  graph 1 nohead lt 1 lw 3 dashtype(3,2) lc "#000000"
set label "Refresh start" left at first refresh_start-3, graph 0.05 rotate by 90 font ",13"
set label "Refresh stop"  left at first refresh_stop-3,  graph 0.05 rotate by 90 font ",13"

plot \
      cmd_filter_dat(dat_file, "True", bucket_sz) using 9:($8 * 1000) w l lw 2 lc rgb "#af2318" t "", \
