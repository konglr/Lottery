
#接下来读取.xls数据
data= xlrd.open_workbook('./快乐8开奖情况.xls')
table = data.sheets()[0]
data_lstm=[]
for i in range(issueCount,0,-1):#在excel中最新的数据在最上面因此要倒序读excel
    x=table.row(i)[2].value
    for j in range(20):
        data_lstm=np.append(data_lstm,float(x[3*j])*10+float(x[3*j+1]))
print(data_lstm)
data_np=data_lstm


# Read the Excel file into a pandas DataFrame
data = pd.read_excel('./快乐8开奖情况.xls')

# Split the numbers in the "开奖号码" column into separate columns
balls_df = data['开奖号码'].str.split(' ', expand=True)

# Rename the columns
balls_df.columns = [f'ball_{i+1}' for i in range(balls_df.shape[1])]

# Concatenate the DataFrame with the issue and open time columns
result_df = pd.concat([data[['期号', '开奖日期', '总销售额(元)']], balls_df], axis=1)

# change the sort of date
result_df = result_df.iloc[::-1]
result_df.reset_index(drop=True, inplace=True)

# change column data Types
result_df['开奖日期'] = pd.to_datetime(result_df['开奖日期'])
result_df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6', 'ball_7', 'ball_8', 'ball_9', 'ball_10',
           'ball_11', 'ball_12', 'ball_13', 'ball_14', 'ball_15', 'ball_16', 'ball_17', 'ball_18', 'ball_19', 'ball_20']] = \
    result_df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6', 'ball_7', 'ball_8', 'ball_9', 'ball_10',
                'ball_11', 'ball_12', 'ball_13', 'ball_14', 'ball_15', 'ball_16', 'ball_17', 'ball_18', 'ball_19', 'ball_20']].astype('int64')


# Display the resulting DataFrame
print(result_df)



# Step 1: Calculate the occurring rate of each number as a percentage
total_draws = len(data_np) // 20  # Total number of draws
number_counts = {}  # Dictionary to store occurrence counts for each number

for number in range(1, 81):
    occurrences = np.count_nonzero(data_np == number)
    occurring_rate = (occurrences / total_draws) * 100  # Convert to percentage
    number_counts[number] = (occurrences, occurring_rate)

# Step 2: Plot the bar chart for occurrence times
plt.figure(figsize=(12, 12))

# Plot the bar chart for occurrence times
plt.subplot(2, 1, 1)
bars = plt.bar(number_counts.keys(), [count[0] for count in number_counts.values()], color='skyblue')
plt.xlabel('Number')
plt.ylabel('Occurrence Times')
plt.title('History Lucky 8 Draw Numbers Occurrence Times')
plt.xticks(np.arange(1, 81, 1))  # Show ticks every numbers
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations for occurrence times to each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

# Plot the line chart for occurring rate
plt.subplot(2, 1, 2)
plt.plot(number_counts.keys(), [count[1] for count in number_counts.values()], color='red', marker='o', linestyle='-')
plt.xlabel('Number')
plt.ylabel('Occurring Rate (%)')
plt.title('History Lucky 8 Draw Numbers Occurring Rate')
plt.xticks(np.arange(1, 81, 1))  # Show ticks every numbers
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations for occurring rate as percentage
for i, rate in enumerate([count[1] for count in number_counts.values()]):
    plt.text(i+1, rate, f'{rate:.2f}%', va='top', ha='center', color='red')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Assuming number_counts is a dictionary
number_counts_df = pd.DataFrame.from_dict(number_counts)

# Print the summary statistics
number_counts_df.iloc[0].describe()
number_counts_df.iloc[1].describe()


#绘制快乐8的一维开奖数据
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams['figure.figsize'] = fig_size
plt.title("happy8 results")
plt.ylabel("Draw results")
plt.xlabel("Data")
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(data_np)
plt.show()