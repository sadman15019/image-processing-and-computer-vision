#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Mon May 15 08:50:35 2023

@author: kanizfatema
"""
degree = {}
def getMembershipStaffing(staff):
   
    if staff < 0 or staff > 100:
        degree["s"] = 0
        degree["l"] = 0
        
    
    
    elif staff <=40:
        degree["s"] = 1
        degree["l"] = 0
        
    
    elif staff>40 and staff<62:
        degree["s"] = float((62-staff)*(1/(62-40)))
        degree["l"] = float((staff-40)*1.0/(62-40))
       
      
    elif staff >=62 and staff <= 100:
        
        degree["s"] = 0
        degree["l"] = 1
        
        
    return degree 

def getMembershipFunding(fund):
    
    if fund < 0 or fund > 100:
        degree["i"] = 0
        degree["m"] = 0
        degree["a"] = 0
    
    
    elif fund <=35:
        degree["i"] = 1
        degree["m"] = 0
        degree["a"] = 0
    
    elif fund>35 and fund<50:
        degree["i"] = float((50-fund)*(1/(50-35)))
        degree["m"] = float((fund-35)*1.0/(50-35))
        degree["a"] = 0
      
    elif fund >=50 and fund <= 70:
        
        degree["i"] = 0
        degree["m"] = 1
        degree["a"] = 0
    
    elif fund>70 and fund<=80:
        degree["m"] = float((80-fund)*(1/(80-70)))
        degree["a"] = float((fund-70)*1.0/(80-70))
        degree["i"] = 0
     
    elif fund>80 and fund<=100:
           degree["m"] = 0
           degree["a"] = 1
           degree["i"] = 0
    return degree




def ruleEvalationAssessment(staff,fund):
    low=0
    normal=0
    high=0
    print(degree)
    low=(max(degree["a"],degree["s"]))
    normal=(min(degree["m"],degree["l"]))
    high=(degree["i"])
    
    return low,normal,high


def defuzzificationAssessment(low,normal,high):
    
    x1=25
    x2=65
    x3=100
    nominator=0
    x=0
    denominator=0
    while x<=100:
        
        if(x>=0 and x<=25 ):
            nominator=nominator+x*low
            denominator=denominator+low
        elif (x>25 and x<=65):  
            nominator=nominator+x*normal
            denominator=denominator+normal
        elif(x>65 and x<=100):
            nominator=nominator+x*high
            denominator=denominator+high
        x+=10
    #cog formula
    cog=nominator/denominator
    return cog

#input
project_funding,project_staffing = 10, 60


fuzzyfunding=getMembershipFunding(project_funding)
fuzzystaffing=getMembershipStaffing(project_staffing)


low,normal,high = ruleEvalationAssessment(project_funding,project_staffing)



conAssessment = defuzzificationAssessment(low,normal,high )
print("Fuzzified Continuous Assessment: ",conAssessment)