#!/usr/bin/env gnuplot

dat_file="data.dat"
set datafile sep '\t'
set output "data.eps"

col_app                          = 1
col_cache_policy                 = 2
col_dataset                      = 3
col_cache_percent                = 4
col_step_time_feat_copy          = 5


set terminal postscript "Helvetica,16" eps enhance color dl 2 background "white"

set pointsize 1
# set size 0.4,0.35
set size 0.85,0.42
# set size 2.4,1.05
# set zeroaxis

set tics font ",14" scale 0.5

# set style data histogram
# set style histogram gap 1
# set style fill solid border -1
# set boxwidth 0.6 relative

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

set multiplot layout 1,2

### Key
set key inside left Left top enhanced nobox reverse # autotitles columnhead
set key samplen 1.5 spacing 1.5 height 0.2 width -2 font ',13' maxrows 4 center at graph 0.7, graph 0.7 #maxrows 1 at graph 0.02, graph 0.975  noopaque
# unset key

## Y-axis
set ylabel "Embedding Extraction\n Time(ms)" offset 2,0
set yrange [0:10]
# set ytics 0,1,5 
set ytics offset 0.5,0 #format "%.1f" #nomirror
set grid y

### X-axis
set xlabel "Cache rate per GPU" offset 0,0.7
# set xrange [0:19]
#set xtics 1,1,8 
set xtics nomirror offset 0,0.3

handle_multiplot_location(row, col) = sprintf("set size 0.4, 0.35; set origin %d*0.4+0.02, 0.05;", col,row)

# set label "{/Helvetica-Italic Server A}" at graph 0.05, graph 0.9 textcolor rgb "#006000"
# set border lc rgb "#006000" lw 2
# set tics textcolor "black"
# eval(handle_multiplot_location(0,0))
# unset xtics

# dat_file="eval-break-tech-4v100.dat"
app="_sup"
dataset="PA"
eval(handle_multiplot_location(0, 0))
plot \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^Rep*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#af2318" t "Rep_U", \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^Cli*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#f19e38" t "Part_U", \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^DIRECTColl*" ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#0080ff" t "+Policy", \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^Coll*" ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 4 lc rgb "#0080ff" t "UGache", \
      # NaN w l lc "white" t " ", \
      # cmd_filter_dat_by_policy(dat_file, app, dataset, "MPSRep*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 4 lc rgb "#af2318" t "Rep+Ext.", \
      # cmd_filter_dat_by_policy(dat_file, app, dataset, "MPSCli*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 4 lc rgb "#f19e38" t "Part+Ext.", \

set yrange [0:40]
unset key
unset ylabel
unset xrange
# set xrange [0:21]
# set xtics 2
# dat_file="eval-break-tech-8a100.dat"
app="_sup"
dataset="CF"
# app="dlrm"
# dataset="SP"
eval(handle_multiplot_location(0, 1))
plot \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^Rep*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#af2318" t "Rep", \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^Cli*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#f19e38" t "Part", \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^DIRECTColl*" ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#0080ff" t "+Plcy.", \
      cmd_filter_dat_by_policy(dat_file, app, dataset, "^Coll*" ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 4 lc rgb "#0080ff" t "UGache", \
      # NaN w l lc "white" t " ", \
      # cmd_filter_dat_by_policy(dat_file, app, dataset, "MPSRep*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 4 lc rgb "#af2318" t "Rep+Ext.", \
      # cmd_filter_dat_by_policy(dat_file, app, dataset, "MPSCli*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 4 lc rgb "#f19e38" t "Part+Ext.", \


# unset key
# unset ylabel
# eval(handle_multiplot_location(0, 1))
# plot \
#       cmd_filter_dat_by_policy("eval-break-portion.dat", "_unsup", "CF", "MPSRep*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#af2318" t "Rep", \
#       cmd_filter_dat_by_policy("eval-break-portion.dat", "_unsup", "CF", "MPSCli*"  ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#f19e38" t "Part", \
#       cmd_filter_dat_by_policy("eval-break-portion.dat", "_unsup", "CF", "MPSColl*" ) using  ((column(col_cache_percent))):(column(col_step_time_feat_copy)*1000) w lp lw 2 lc rgb "#0080ff" t "UGache", \
