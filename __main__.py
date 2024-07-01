from deepface import DeepFace
import matplotlib.pyplot as plt

# dfs = DeepFace.find(
#   img_path = "photo_2024-04-16_11-54-40.jpg",
#  db_path = ".",
# )

face_objects = DeepFace.extract_faces(img_path='project_data/inavlids.jpg')
# print(face_objects[0])

# face_objects2 = DeepFace.extract_faces(img_path='3shi.jpg', enforce_detection=False)

# plt.imshow(face_objects[0]['face'])
# plt.show()
for f_obj in face_objects:
    plt.imshow(f_obj['face'])
    plt.show()

# for f_obj in face_objects2:
#    plt.imshow(f_obj['face'])
#    plt.show()
