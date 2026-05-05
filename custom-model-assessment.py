
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

def customAssesment2(labels,preds):
    punishmentForMiss = -1
    punishmentForFalsePositive = -20
    rewardForHit = 10
    score = 0
    allToHit = 0
    allPos = 0
    hit = 0
    for i in range(len(labels)):
        points = 0
        if preds[i] == 1:
            allPos += 1
        if labels[i] == 1 and preds[i] == 1:
            points += 1
            allToHit += 1
            hit +=1
        if labels[i] == 1 and preds[i] == 0:
            allToHit += 1
        score += points
    print(f"custom assesment 2: \n...{allToHit} to hit, \n...{allPos} predicted positive, \n...{hit} of them correctly")
    return score

def customAssesment3(labels,preds):
    punishmentForMiss = -1
    punishmentForFalsePositive = -20
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