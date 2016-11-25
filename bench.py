#!/usr/bin/python
#     This file is part of CUDA_eternity2
#     Copyright (C) 2016  Julien Thevenon ( julien_thevenon at yahoo.fr )
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>
#
import subprocess
import datetime
import sys

def run_process(nb_cases, nb_block, block_size):
        p = subprocess.Popen(["/usr/bin/time", "./bin/CUDA_eternity2.exe", "--nb_cases=" + str(nb_cases), "--nb_block=" + str(nb_block), "--block_size=" + str(block_size)],stderr=subprocess.PIPE)
        for line in p.stderr:
                index = line.find("elapsed")
                if -1 <> index:
                        line = line[0:index]
                        index = line.find("system ")
                        line = line[index+7:]
                        time_tuple = datetime.datetime.strptime(line, "%M:%S.%f").time()
                        return time_tuple

# Perform several executions and return average execution time in seconds
def average_runtime(iteration_nb,nb_cases, nb_block, block_size):
        print '--------------------'
        elapsed_deltas = []
        # Perform all executions and store elapsed_time in a list
        for i in range(0,iteration_nb):
                print "RUN ", i
                elapsed_time = run_process(nb_cases,nb_block,block_size)
                elapsed_delta = datetime.timedelta(hours=elapsed_time.hour,minutes=elapsed_time.minute,seconds=elapsed_time.second,microseconds=elapsed_time.microsecond)
                elapsed_deltas.append(elapsed_delta)

        # compute average time
        mean_time = datetime.timedelta(hours=0,minutes=0,seconds=0,microseconds=0)
        for elapsed_delta in elapsed_deltas:
                mean_time = mean_time + elapsed_delta
        mean_time = mean_time / iteration_nb
        return (1000000 * mean_time.seconds + mean_time.microseconds) / 1000000.0


print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
nb_iteration = 10
nb_cases = 1048576
if len(sys.argv) >=2:
        nb_cases = int(sys.argv[1])
if len(sys.argv) >=3:
        nb_iteration = int(sys.argv[2])
print "Nb cases : " + str(nb_cases)
print "Nb iterations : " + str(nb_iteration)

iteration_list = [ 2 ** x for x in range(11)]

report = open("report.csv","w")
report.write("Bench results:\n")
report.write("Nb block/Block size")
for block_size in iteration_list:
        report.write("," + str(block_size))
report.write("\n")
for nb_block in iteration_list:
        report.write(str(nb_block))
        for block_size in iteration_list:
                report.write("," + str(average_runtime(nb_iteration,nb_cases,nb_block,block_size)))
        report.write("\n")
report.close()

print "Report generated !"
#EOF
