import pandas as pd

def find_students_with_min_score(filepath, target_score=45):
    # Load data 
    data = pd.read_excel(filepath)
    
    # Specify the exact names of the lab columns
    lab_columns = ['Lab 8 (342508)', 'Lab 9 (342509)', 'Lab 10 (342497)', 'Lab 11 (342498)', 'Lab 12 (342499)']
    
    # Convert lab score columns to numeric, handling non-numeric values by converting them to NaN
    data[lab_columns] = data[lab_columns].apply(pd.to_numeric, errors='coerce')
    
    # Calculate the total score for each student across the specified labs
    data['Total Score'] = data[lab_columns].sum(axis=1)
    
    # Filter students whose total score is at least 45 and select their ANON_ID as integers
    qualifying_students = data[data['Total Score'] >= target_score]['ANON_ID'].dropna().astype(int)

    return qualifying_students


if __name__ == "__main__":
    filepath = 'FinalGrades.xlsx'  
    result = find_students_with_min_score(filepath)
    high_students = []
    count = 0
    print("ANON_IDs with a total score of at least 49/50 from Labs 8-12:")
    for anon_id in result:
        count+=1
        high_students.append(anon_id) 
        print(anon_id)  
    print(count)
    print(high_students)