#!/usr/bin/env gnuplot

dat_file="data.dat"
set datafile sep '\t'
set output "data.eps"

col_app                          = 1
col_cache_policy                 = 2
col_dataset                      = 3
col_cache_percent                = 4
col_Wght_L                       = 5
col_Wght_R                       = 6
col_Wght_C                       = 7



set terminal postscript "Helvetica,16" eps enhance color dl 2 background "white"

set pointsize 1
set size 0.82,0.75

set tics font ",14" scale 0.5

set style fill solid border -1

format_str="<awk -F'\\t' '{ if(". \
                              "$".col_app."           ~ \"%s\"     && ". \
                              "$".col_dataset."       ~ \"%s\"     && ". \
                              "$".col_cache_policy."  ~ \"%s\"     ". \
                              ") { print }}' %s "
cmd_filter_dat_by_policy(file,app,dataset,policy)=sprintf(format_str, app, dataset, policy, file)

set rmargin 0 #2
set lmargin 5 #5.5
# set tmargin 0.5 #1.5
set tmargin 0.5 #1.5
set bmargin 1 #2.5

set multiplot layout 2,1

### Key
set key outside left Right top enhanced nobox horizontal # autotitles columnhead
set key samplen 1.5 spacing 1.5 height 0.2 width 0 maxcols 3 font ',13' center at graph 0.45, graph 1.15 #maxrows 1 at graph 0.02, graph 0.975  noopaque
# unset key

## Y-axis
set ylabel "Percent from source" offset 2.4,-5
set yrange [0:100]
# set ytics 0,1,5 
set ytics 0,20,100 offset 0.5,0 #format "%.1f" #nomirror
set grid y

set xrange [1:13]
set xtics nomirror offset 0,0.3

handle_multiplot_location(row, col) = sprintf("set size 0.8, 0.33; set origin 0, 0.67-(%d+1)*0.31;", row)

unset xtics
set boxwidth 0.3
handle_x_offset(orig, offset)=orig + offset * 0.4
eval(handle_multiplot_location(0, 0))
plot \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "MPSPhaseCli*") using  (handle_x_offset(column(col_cache_percent), -1)):(column(col_Wght_L) + column(col_Wght_R) + column(col_Wght_C))  w boxes lc rgb "#f19e38" fs solid 0.2  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "MPSPhaseCli*") using  (handle_x_offset(column(col_cache_percent), -1)):(column(col_Wght_L) + column(col_Wght_R))                       w boxes lc rgb "#f19e38" fs solid 0.6  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "MPSPhaseCli*") using  (handle_x_offset(column(col_cache_percent), -1)):(column(col_Wght_L))                                            w boxes lc rgb "#f19e38" fs solid 1.0  t "Part_U", \
      NaN w boxes lc rgb "#000000" fs solid 1.0 t "From Local",      \
      \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "^Coll*") using  (handle_x_offset(column(col_cache_percent),  0)):(column(col_Wght_L) + column(col_Wght_R) + column(col_Wght_C)) w boxes lc rgb "#0080ff" fs solid 0.2 t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "^Coll*") using  (handle_x_offset(column(col_cache_percent),  0)):(column(col_Wght_L) + column(col_Wght_R))                      w boxes lc rgb "#0080ff" fs solid 0.6 t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "^Coll*") using  (handle_x_offset(column(col_cache_percent),  0)):(column(col_Wght_L))                                           w boxes lc rgb "#0080ff" fs solid 1.0 t "UGache", \
      NaN w boxes lc rgb "#000000" fs solid 0.6 t "From Remote GPU", \
      \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "MPSPhaseRep*") using  (handle_x_offset(column(col_cache_percent),  1)):(column(col_Wght_L) + column(col_Wght_R) + column(col_Wght_C))  w boxes lc rgb "#af2318" fs solid 0.2  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "MPSPhaseRep*") using  (handle_x_offset(column(col_cache_percent),  1)):(column(col_Wght_L) + column(col_Wght_R))                       w boxes lc rgb "#af2318" fs solid 0.6  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "PA", "MPSPhaseRep*") using  (handle_x_offset(column(col_cache_percent),  1)):(column(col_Wght_L))                                            w boxes lc rgb "#af2318" fs solid 1.0  t "Rep_U", \
      NaN w boxes lc rgb "#000000" fs solid 0.2 t "From Host DRAM"

unset key
unset ylabel
set xlabel "Cache rate per GPU" offset 0,0.7
set xtics nomirror offset 0,0.3
eval(handle_multiplot_location(1, 0))
# set output "eval-break-portion-less-skew.eps"
plot \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "MPSPhaseCli*") using  (handle_x_offset(column(col_cache_percent), -1)):(column(col_Wght_L) + column(col_Wght_R) + column(col_Wght_C))  w boxes lc rgb "#f19e38" fs solid 0.2  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "MPSPhaseCli*") using  (handle_x_offset(column(col_cache_percent), -1)):(column(col_Wght_L) + column(col_Wght_R))                       w boxes lc rgb "#f19e38" fs solid 0.6  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "MPSPhaseCli*") using  (handle_x_offset(column(col_cache_percent), -1)):(column(col_Wght_L))                                            w boxes lc rgb "#f19e38" fs solid 1.0  t "Part", \
      \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "^Coll*") using  (handle_x_offset(column(col_cache_percent),  0)):(column(col_Wght_L) + column(col_Wght_R) + column(col_Wght_C)) w boxes lc rgb "#0080ff" fs solid 0.2 t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "^Coll*") using  (handle_x_offset(column(col_cache_percent),  0)):(column(col_Wght_L) + column(col_Wght_R))                      w boxes lc rgb "#0080ff" fs solid 0.6 t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "^Coll*") using  (handle_x_offset(column(col_cache_percent),  0)):(column(col_Wght_L))                                           w boxes lc rgb "#0080ff" fs solid 1.0 t "UGache", \
      \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "MPSPhaseRep*") using  (handle_x_offset(column(col_cache_percent),  1)):(column(col_Wght_L) + column(col_Wght_R) + column(col_Wght_C))  w boxes lc rgb "#af2318" fs solid 0.2  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "MPSPhaseRep*") using  (handle_x_offset(column(col_cache_percent),  1)):(column(col_Wght_L) + column(col_Wght_R))                       w boxes lc rgb "#af2318" fs solid 0.6  t "", \
      cmd_filter_dat_by_policy(dat_file, "_sup", "CF", "MPSPhaseRep*") using  (handle_x_offset(column(col_cache_percent),  1)):(column(col_Wght_L))                                            w boxes lc rgb "#af2318" fs solid 1.0  t "Rep"
