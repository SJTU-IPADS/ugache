#!/usr/bin/env gnuplot

dat_file="data.dat"
set datafile sep '\t'
set output "data.eps"

col_app                          = 1
col_cache_policy                 = 2
col_dataset                      = 3
col_step_time_feat_copy           = 4



set terminal postscript "Helvetica,16" eps enhance color dl 2 background "white"

set pointsize 1
# set size 0.4,0.35
set size 0.66,0.45
# set size 2.4,1.05
# set zeroaxis

set tics font ",14" scale 0.5

set style data histogram
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.6 relative

format_str="<awk -F'\\t' '{ if(". \
                              "$".col_app."           ~ \"%s\"     && ". \
                              "$".col_cache_policy."  ~ \"%s\"     ". \
                              ") { print }}' %s "
                              # ") { print; print >> 11 }}' %s "
cmd_filter_dat_by_policy(file,app,policy)=sprintf(format_str, app, policy, file)

set rmargin 0 #2
set lmargin 5 #5.5
# set tmargin 0.5 #1.5
set tmargin 0.5 #1.5
set bmargin 1 #2.5

### Key
set key inside left Right top enhanced nobox # autotitles columnhead
set key samplen 1.5 spacing 1.5 height 0.2 width 0 font ',13' center at graph 0.1, graph 0.8 #maxrows 1 at graph 0.02, graph 0.975  noopaque
unset key

set multiplot layout 3,3

## Y-axis
set yrange [0:]
# set ytics 0,1,5 
set ytics offset 0.5,0 #format "%.1f" #nomirror
set grid y

### X-axis
#set xlabel "Number of GPUs" offset 0,0.7
# set xrange [-5:5]
#set xtics 1,1,8 
set xtics nomirror offset 0,0.3

# fs_pattern_A = 6
# fs_pattern_B = 7
# fs_pattern_C = 2
fs_pattern_A = 3
fs_pattern_B = 3
fs_pattern_C = 3

handle_multiplot_location(row, col) = sprintf("set size 0.33, 0.33; set origin %d*0.32, 0.07;", col,row)

set label "{/Helvetica-Italic Server A}" at graph 0.05, graph 0.9 textcolor rgb "#006000"
set border lc rgb "#006000" lw 2
set tics textcolor "black"

set ytics offset 0.5,0 #format "%.1f" #nomirror
unset xtics

eval(handle_multiplot_location(0,0))
set arrow from  graph 0.2, graph -0.17 to graph 0.8, graph -0.17 nohead lt 1 lw 2 lc "#000000" front
set yrange [0:60]
set tics textcolor "black"
set xtics nomirror offset 0,0.3
set xlabel "GNN Sup." offset 0,0.3
plot cmd_filter_dat_by_policy(dat_file, "sage_sup", "^Rep"            ) using (1000*(column(col_step_time_feat_copy))):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#af2318" t "Rep", \
     cmd_filter_dat_by_policy(dat_file, "sage_sup", "^Cliq"           ) using (1000*(column(col_step_time_feat_copy))):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#f19e38" t "Part", \
     cmd_filter_dat_by_policy(dat_file, "sage_sup", "^Coll*"          ) using (1000*(column(col_step_time_feat_copy))):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#0080ff" t "UGache", \

unset label



eval(handle_multiplot_location(0,1))
set xlabel "GNN Unsup." offset 0,0.3
set yrange [0:100]
plot cmd_filter_dat_by_policy(dat_file, "sage_unsup", "^Rep"          ) using (1000*(column(col_step_time_feat_copy))):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#af2318" t "", \
     cmd_filter_dat_by_policy(dat_file, "sage_unsup", "^Cliq"         ) using (1000*(column(col_step_time_feat_copy))):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#f19e38" t "", \
     cmd_filter_dat_by_policy(dat_file, "sage_unsup", "^Coll*"        ) using (1000*(column(col_step_time_feat_copy))):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#0080ff" t "", \

unset xlabel


set multiplot previous
eval(handle_multiplot_location(0,1))
unset grid
set key outside left Right top enhanced nobox # autotitles columnhead
unset xtics
# width 0 or 4?
set key samplen 3 spacing 0 height 3 width 3 font ',16' maxrows 1 center at graph -0.3, graph 1.1 #maxrows 1 at graph 0.02, graph 0.975  noopaque
plot \
     NaN w histogram fs lc rgb "#af2318" t "Rep_U", \
     NaN w histogram fs lc rgb "#f19e38" t "Part_U", \
     NaN w histogram fs lc rgb "#0080ff" t "UGache" \

     # NaN w histogram fs pattern 2 lc rgb "#af2318" t "HPS", \
     # NaN w histogram fs pattern 2 lc rgb "#f19e38" t "SOK", \

