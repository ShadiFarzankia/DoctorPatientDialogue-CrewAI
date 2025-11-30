we have a dataset of doctor-patient dialog with the structure in the image. i also have have these live project about the doctor patient dialogs and i want to implement these part as a crewai project. you can see the structure of my project. from what we talked before, i am not sure if we need to write code or prompt for having this. give me the steps and codes or whatever. we dont want the exact thing in the living project. because in that, it is recording voice of dialog between patient and doctor. but in our project we want to have this scenario: our agents are: 
1. one agent for accurate syptoms 
2. one agent for drug history 
3. another for orchestration and supervising 
4. we have 5000 doctor-patinet dialogs data that is in the link below: 
- Training Data:
https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data_json/train_full.json

- Validation Data:
https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data_json/valid_full.json

- Test Data:
https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data_json/clinicalnlp_taskB_test1_full.json

i am not sure if we need test and train data or not. but if needed, use these datasets above.

in the ui page that patient can  enter his/her patient complaint(and symptoms). after pressing the enter, in the next step, system should analysis the text of the patient and with considering accurate syptoms part of the 5000 data that we trained our LLM, it should pick the potential symptoms based on the similarity of the patinet syptoms and top 5 closest syptoms of our dialogs suggest to the patient other syptoms as checkbox. patinet will check some of the checkboxes or a checkbox as a none of the syptoms. and press the enter. in the next page, other agent should also extract the related drugs to the syptoms similar to our patinet and suggest used drugs for the patinet as a checkbox and ask which of these drugs the patinet is using. in the next step, patinet will check some of the drugs that is using or used in the passed from the checkboxes or if the patinet didnt use any of the drugs, he checks the a checkbox as a none of the drugs. and press the enter. after that, ih the summarization part, the patinet will see a summary that what he should do for the next step(suggest a step using the onjective part: phisical examination, radilogy, lab eximation or suggest that he needs to reserve a doctor visit time or whatever that agant can find suitable from 5000 data for pationt) 
with considering subjective, objective agents, design this project.