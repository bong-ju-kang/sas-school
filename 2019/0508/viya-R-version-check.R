# Bong Ju Kang
# for version check
# 5/8/2019

R.version.string

pkglist <- installed.packages()
colnames(pkglist)
rownames(pkglist)
print(pkglist[,c('Package', 'Version')])