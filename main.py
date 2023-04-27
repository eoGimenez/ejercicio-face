import os
import cv2
import face_recognition as freco
import numpy
from datetime import datetime


path_imgs = 'Empleados'
employees = []
employees_name = []
employees_array = os.listdir(path_imgs)

for name in employees_array:
    img = cv2.imread(f'{path_imgs}/{name}')
    employees.append(img)
    employees_name.append(os.path.splitext(name)[0])


def codificate(images):
    transformed_array = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        code_img = freco.face_encodings(img)[0]
        transformed_array.append(code_img)

    return transformed_array


def check_in(person):
    with open('register.cvs', "r+") as new_entry:
        info_array = new_entry.readlines()
        register_name = []
        for line in info_array:
            add = line.split(',')
            register_name.append[add[0]]

        if person not in register_name:
            now = datetime.now()
            formate_now = now.strftime('%H:%M:%S')
            new_entry.writelines(f'\n{person}, {formate_now}')


transformed_employees_array = codificate(employees)

# take pic from webcam

shoot = cv2.VideoCapture(0, cv2.CAP_DSHOW)

success, img = shoot.read()

if not success:
    print('Saque una nueva foto')
else:
    face_img = freco.face_locations(img)

    face_codificated = freco.face_encodings(img, face_img)

    for codificated_face, ubication_face in zip(face_codificated, face_img):
        same_face = freco.compare_faces(
            transformed_employees_array, codificated_face)
        distance = freco.face_distance(
            transformed_employees_array, codificated_face)
        success_index = numpy.argmin(distance)
        if distance[success_index] > 0.6:
            print('No coincide con ningun empleado')
        else:
            name = employees_name[success_index]
            y1, x2, y2, x1 = ubication_face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

            check_in(name)

            cv2.imshow('Imagen web', img)
            cv2.waitKey(0)
