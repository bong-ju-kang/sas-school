# Bong Ju Kang
# for version check
# 5/8/2019

import sys

print(sys.version)

import pkg_resources
l = [(p.project_name, p.version) for p in pkg_resources.working_set]
for p, v in sorted(l):
	print(p, v)

# 내부 변수 확인
print('dm_partitionvar=', dm_partitionvar)
print('dm_partition_train_val=', dm_partition_train_val)

# _PartInd_ 값 확인
print('dm_inputdf: _PartInd_ value=', dm_inputdf['_PartInd_'].value_counts())
