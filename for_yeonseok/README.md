1. belief_state.json을 simmc/data/simmc_furniture/ 안으로 옮긴다 
2. belief_state.json 을 데이터 로드하는 과정에서 같이 load 하여서 batch 에서 함께 불러올 수 있도록 코드를 수정한다. 
3. 본 branch 에 commit 하여 push 한다. 

* 아마 belief_state 의 turn index와 batch의 dialog - turn index 를 맞추는 부분이 조금 까다로울 것임(아까 우리가 말했던 것 처럼 len 정보를 사용해서 padding 부분을 맞춰서 넣어주면 될 것)

화이팅 ! : )
