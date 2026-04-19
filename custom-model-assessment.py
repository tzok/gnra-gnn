
def customAssesment1(labels,preds):
    punishmentForMiss = -20
    punishmentForFalsePositive = -7
    rewardForHit = 10
    score = 0
    for i in range(len(labels)):
        points = 0
        if labels[i] == 1 and preds[i] == 1:
            points += rewardForHit*2
        if labels[i] != preds[i] and labels[i] == 1:
                #we either punish a miss, or reward a close call 
                points = punishmentForMiss
                #check the neighborhood for positive labels
                if i> 1 and labels[i-2] == 1:
                    points = rewardForHit*0.5
                if i < len(labels)-2 and labels[i+2] == 1:
                    points = rewardForHit*0.5
                if i > 0 and labels[i-1] == 1:
                    points = rewardForHit*0.75
                if i < len(labels)-1 and labels[i+1] == 1:
                    points = rewardForHit*0.75
                
        if labels[i] == 0 and preds[i] == 1:
            points += punishmentForFalsePositive
            #check the neighborhood for positive labels
            if i> 1 and labels[i-2] == 1:
                points = rewardForHit*0.5
            if i < len(labels)-2 and labels[i+2] == 1:
                points = rewardForHit*0.5
            if i > 0 and labels[i-1] == 1:
                points = rewardForHit*0.75
            if i < len(labels)-1 and labels[i+1] == 1:
                points = rewardForHit*0.75
        score += points
    return score