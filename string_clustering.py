from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

class StringClustering:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.common_substring = ""

    def longest_common_substring(self, strings):
        if not strings:
            return ""
        substr = strings[0]
        for s in strings[1:]:
            temp_substr = ""
            for i in range(len(substr)):
                for j in range(i + 1, len(substr) + 1):
                    if substr[i:j] in s and len(substr[i:j]) > len(temp_substr):
                        temp_substr = substr[i:j]
            substr = temp_substr
        return substr

    def remove_common_substring(self, strings):
        self.common_substring = self.longest_common_substring(strings)
        print("Common Substring:", self.common_substring)
        return [s.replace(self.common_substring, "").strip() for s in strings]

    def cluster_strings(self, strings):
        cleaned_strings = self.remove_common_substring(strings)
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
        X = vectorizer.fit_transform(cleaned_strings)
        
        clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', 
                                             distance_threshold=self.threshold)
        labels = clustering.fit_predict(X.toarray())
        
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(strings[i])
            
        clusters = {int(label): group for label, group in clusters.items()}
        
        return clusters

# # Example usage
# strings = [
#     "Screw, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
#     "Screw, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
#     "Screw, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
#     "Screw, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
#     "Screw, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
#     "M4 Insert, IP67 Enclosure, 130 mm x 80 mm x 40 mm_default",
#     "M4 Insert, IP67 Enclosure, 130 mm x 80 mm x 40 mm_default",
#     "Seal, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
#     "Cover, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
#     "Enclosure, IP67 Enclosure, 130 mm x 80 mm x 40 mm",
# ]

# helper = StringClustering(threshold=0.8)
# clusters = helper.cluster_strings(strings)

# print("Common Substring:", helper.common_substring)
# print("Clusters:")
# for label, group in clusters.items():
#     print(f"Cluster {label}: {group}")