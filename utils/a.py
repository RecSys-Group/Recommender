
class feature:
    def __init__(self, n, l):
        self.name = n
        self.content = l

    def match(self, otherFeature):
        rep = []
        for i in otherFeature.content:
            if i in self.content:
                rep.append(i)
        return  self.name, otherFeature.name, rep, len(rep)

if __name__ == "__main__":
    features = []
    beauty = feature('beauty', ['hair', 'staff', 'job', 'massage', 'service', 'room', 'salon', 'nails', 'spa', 'location'])
    features.append(beauty)
    entertainment = feature('entertainment', ['show', 'room', 'rooms', 'seats', 'staff', 'hotel', 'theater', 'location', 'bar', 'tickets'])
    features.append(entertainment)
    food = feature('food', ['food', 'service', 'pizza', 'taste', 'cheese', 'location', 'staff', 'flavor', 'fries', 'prices'])
    features.append(food)
    health = feature('health', ['staff', 'care', 'appointment', 'patient', 'doctor', 'location', 'office', 'paperwork', 'dentist', 'pain'])
    features.append(health)
    yelp_clothing = feature('yelp_clothing', ['store', 'selection', 'prices', 'staff', 'deals', 'shoes', 'price', 'quality', 'clothes', 'items'])
    features.append(yelp_clothing)
    bars = feature('bars', ['food', 'bar', 'service', 'def', 'wait', 'drinks', 'hour', 'music', 'staff', 'table'])
    features.append(bars)

    n = len(features)
    all = set()
    for i in range(n):
        for k in features[i].content:
            all.add(k)
        for j in range(i+1, n):
            print features[i].match(features[j])

    print 'all feature',len(all)