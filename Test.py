# API text file
KEY_FILE = 'API.txt'
with open(KEY_FILE, "r") as f:
    key = f.readline().strip()

query = """
query GetMeasurePopulationData {
  measure_A(metricId: 171) {
    measureId
    name
    description
    unitType
    format
    source {
      name
    }
    subpopulations {
      name
      population {
        name
        populationCategory {
          name
        }
      }
      data(where: { state: { in: ["ALL"] } }) {
        endDate
        dateLabel
        value
      }
    }
  }
}
"""

url = 'https://api.americashealthrankings.org/graphql'
headers = {
    'Content-Type': 'application/json',
    'X-Api-Key': key
}

response = requests.post(url, json={'query': query}, headers=headers)
print(response.json())
if response.status_code != 200:
    raise Exception(f"GraphQL API request failed with status {response.status_code}:\n{response.text}")
