fcm=read.csv("D:/R/fcm2.csv")
shapiro.test(fcm[which(fcm$Group==1),2])
for (i in 2:10)
  cat(colnames(fcm)[i],"p=",t.test(fcm[which(fcm$Group==0),i],fcm[which(fcm$Group==1),i])$p.value,"\n")

fcm[which(fcm$Group==1),1]

fcm[which(fcm$Group==1),2]
fcm[which(fcm$Group==0),2]
