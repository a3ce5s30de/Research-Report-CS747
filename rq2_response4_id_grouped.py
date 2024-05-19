import pandas as pd
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models


nltk_download('punkt')
nltk_download('stopwords')

# Load data
def load_data(filepath, id_column='ANON_ID', column_name='Response 4', filter_ids=None):
    """ Load the data and filter for specific ANON_IDs if provided """
    data = pd.read_excel(filepath)
    data = data[[id_column, column_name]].dropna()
    data[id_column] = data[id_column].astype(int)  # Convert ANON_IDs to integer to remove decimals
    if filter_ids is not None:
        data = data[data[id_column].isin(filter_ids)]
    return data

# Preprocess the textual data
def preprocess_data(texts):
    """ Tokenize, remove stopwords, and lemmatize the text """
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_texts = []
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        preprocessed_texts.append(cleaned_tokens)
    return preprocessed_texts

# Build LDA model
def build_lda_model(texts, num_topics=10, seed=100):
    """ Build and return an LDA model with a consistent random seed """
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=seed)
    return lda_model, corpus, dictionary

# Visualize topics
def visualize_topics(lda_model, corpus, dictionary):
    """ Generate HTML visualization of the topics """
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, 'lda.html')

# Classify responses and group by dominant topic
def classify_responses(model, dictionary, new_data):
    """ Classify responses based on the dominant topic and group ANON_IDs by topic """
    processed_responses = [preprocess_data([response]) for response in new_data['Response 4']]
    bow_corpus = [dictionary.doc2bow(text[0]) for text in processed_responses]

    topics = []
    for doc in bow_corpus:
        topic_probs = model.get_document_topics(doc)
        dominant_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0][0]
        topics.append(dominant_topic)

    new_data['Dominant Topic'] = topics
    grouped_data = new_data.groupby('Dominant Topic')['ANON_ID'].apply(list).to_dict()
    
    print(grouped_data)
    topic_items = {topic: len(responses) for topic, responses in grouped_data.items()}

    # Calculate the total number of items (responses) across all topics
    total_items = sum(topic_items.values())

    # Print the number of items (responses) for each topic and the total number of items
    for topic, num_items in topic_items.items():
        print(f"{topic}: {num_items}")

    print(f"Total items across all topics: {total_items}")

    # Sort the topic_items dictionary by the number of items in each topic in descending order
    sorted_topics = sorted(topic_items.items(), key=lambda x: x[1], reverse=True)

    # Select the top 5 topics
    top_5_topics = sorted_topics[:5]

    # Print the top 5 topics
    for topic, num_items in top_5_topics:
        print(f"{topic}: {num_items}")


def main():
    filepath = 'Lab7Responses.xlsx'
    specific_ids = [1498, 1586, 1822, 1312, 1652, 1127, 1405, 1400, 1548, 1055, 1165, 1028, 1667, 1449, 1167, 1499, 1301, 1836, 1651, 1564, 1616, 1029, 1860, 1532, 1439, 1136, 1230, 1418, 1295, 1875, 1797, 1073, 1285, 1210, 1169, 1731, 1383, 1591, 1392, 1252, 1185, 1588, 1196, 1110, 1785, 1662, 1047, 1362, 1780, 1172, 1853, 1511, 1415, 1653, 1687, 1257, 1072, 1734, 1176, 1808, 1002, 1353, 1493, 1654, 1303, 1621, 1296, 1345, 1512, 1420, 1122, 1014, 1581, 1577, 1282, 1346, 1374, 1638, 1099, 1817, 1754, 1211, 1895, 1416, 1148, 1784, 1529, 1177, 1069, 1583, 1882, 1164, 1597, 1198, 1159, 1777, 1049, 1199, 1399, 1489, 1898, 1861, 1848, 1307, 1824, 1614, 1668, 1326, 1298, 1429, 1089, 1881, 1756, 1736, 1692, 1878, 1676, 1696, 1339, 1143, 1543, 1515, 1240, 1768, 1488, 1772, 1357, 1863, 1389, 1471, 1659, 1422, 1100, 1278, 1888, 1582, 1774, 1067, 1015, 1566, 1729, 1139, 1251, 1075, 1232, 1381, 1132, 1052, 1442, 1623, 1311, 1801, 1660, 1451, 1633, 1514, 1584, 1530, 1376, 1351, 1700, 1001, 1030, 1611, 1363, 1408, 1886, 1522, 1508, 1707, 1521, 1050, 1088, 1464, 1889, 1168, 1473, 1846, 1238, 1737, 1769, 1090, 1486, 1011, 1681, 1880, 1745, 1670, 1206, 1141, 1266, 1221, 1795, 1710, 1639, 1147, 1082, 1031, 1237, 1133, 1689, 1460, 1235, 1209, 1743, 1175, 1119, 1162, 1643, 1606, 1806, 1851, 1669, 1528, 1764, 1048, 1138, 1087, 1716, 1270, 1492, 1538, 1815, 1144, 1338, 1062, 1156, 1194, 1733, 1788, 1192, 1721, 1217, 1765, 1672, 1283, 1719, 1094, 1025, 1294, 1008, 1832, 1685, 1724, 1829, 1501, 1314, 1647, 1027, 1571, 1753, 1191, 1443, 1256, 1644, 1292, 1841, 1070, 1414, 1403, 1348, 1550, 1823, 1467, 1245, 1438, 1042, 1640, 1810, 1842, 1291, 1506, 1242, 1849, 1845, 1003, 1061, 1320, 1866, 1213, 1899, 1599, 1723, 1425, 1287, 1741, 1819, 1246, 1219, 1181, 1746, 1224, 1397, 1634, 1423, 1854, 1712, 1366, 1600, 1288, 1630, 1595, 1281, 1380, 1813, 1751, 1835, 1708, 1349, 1409, 1728, 1375, 1604, 1632, 1018, 1344, 1699, 1431, 1559, 1043, 1540, 1683, 1718, 1749, 1858, 1041, 1365, 1108, 1608, 1364, 1504, 1666, 1180, 1873, 1264, 1585, 1347, 1642, 1016, 1093, 1333, 1342, 1045, 1306, 1183, 1249, 1798, 1698, 1645, 1686, 1594, 1332, 1059, 1533, 1762, 1840, 1495, 1253, 1703, 1526, 1161, 1137, 1787, 1837, 1432, 1358, 1279, 1131, 1021, 1589, 1568, 1254, 1450, 1125, 1178, 1726, 1330, 1494, 1404, 1269, 1193, 1271, 1445, 1382, 1244, 1563, 1476, 1760, 1587, 1318, 1260, 1379, 1286, 1343, 1612, 1487, 1155, 1321, 1071, 1809, 1066, 1761, 1783, 1033, 1544, 1359, 1114, 1619, 1679, 1755, 1545, 1179, 1510, 1435, 1354, 1000, 1079, 1402, 1636, 1800, 1328, 1203, 1455, 1477, 1649, 1310, 1105, 1239, 1065, 1299, 1086, 1838, 1735, 1273, 1714, 1590, 1773, 1739, 1883, 1553, 1821, 1195, 1620, 1352, 1485, 1637, 1214, 1661, 1074, 1771, 1820, 1629, 1395, 1624, 1664, 1401, 1865, 1793, 1570, 1715, 1517, 1305, 1184, 1104, 1426, 1518, 1827, 1369, 1189, 1847, 1794, 1109, 1377, 1452, 1440, 1329, 1158, 1076, 1593, 1890, 1309, 1140, 1216, 1483, 1171, 1182, 1807, 1747, 1078, 1444, 1218, 1039, 1135, 1037, 1398, 1268, 1116, 1826, 1255, 1044, 1390, 1658, 1573, 1280, 1267, 1241, 1146, 1313, 1272, 1598, 1091, 1870, 1626, 1436, 1805, 1022, 1355, 1816, 1799, 1549, 1372, 1308, 1694, 1331, 1546, 1417, 1262, 1603, 1325, 1368, 1411, 1556, 1160, 1782, 1831, 1461, 1412, 1618, 1322, 1631, 1259, 1613, 1023, 1677, 1275, 1732, 1447, 1187, 1825, 1010, 1053, 1742, 1693, 1046, 1579, 1770, 1513, 1407, 1188, 1428, 1490, 1166, 1864, 1197, 1419, 1609, 1370, 1592, 1304, 1792, 1223, 1839, 1547, 1551, 1228, 1523, 1850, 1857, 1856, 1717, 1434, 1393, 1229, 1083, 1396, 1859, 1500, 1565, 1036, 1896, 1779, 1778, 1124, 1226, 1026, 1017, 1054, 1738, 1468, 1580, 1387, 1567, 1173, 1145, 1101, 1371, 1174, 1250, 1675, 1421, 1516, 1258, 1887, 1327, 1430, 1722, 1725, 1356, 1261, 1811, 1869, 1384, 1655, 1248, 1834, 1868, 1222, 1297, 1706, 1537, 1682, 1441, 1437, 1657, 1470, 1601, 1674, 1802, 1406, 1233, 1557, 1006, 1830, 1004, 1894, 1690, 1578, 1097, 1465, 1092, 1225, 1697, 1424, 1316, 1748, 1265, 1872, 1555, 1410, 1507, 1300, 1531, 1200, 1505, 1098, 1433, 1558, 1386, 1818, 1336, 1466, 1293, 1602, 1009, 1812, 1157, 1142, 1713, 1617, 1019, 1007, 1024, 1085, 1284, 1186, 1503, 1720, 1113, 1705, 1163, 1323, 1103, 1554, 1650, 1207, 1479]
    responses_data = load_data(filepath, filter_ids=specific_ids)
    preprocessed_texts = preprocess_data(responses_data['Response 4'])
    lda_model, corpus, dictionary = build_lda_model(preprocessed_texts, seed=49)  # Added seed for consistency
    visualize_topics(lda_model, corpus, dictionary)

    # Use the LDA model and dictionary to classify responses
    classify_responses(lda_model, dictionary, responses_data)

if __name__ == "__main__":
    main()
