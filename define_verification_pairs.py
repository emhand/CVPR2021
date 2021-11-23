import os


def ourCar():
    images = []
    if os.path.exists("./namesLists/our_car_imageslist.txt") == False:
        for root, dirs, files in os.walk("./our_car/new-cropped/"):
            if root != "./our_car/new-cropped":
                person = root.replace("./our_car/new-cropped/", "")
                for file in files:
                    image = person + "/" + file
                    images.append(image)
        images = sorted(images)
        imout = open("./namesLists/our_car_imageslist.txt", "w")
        for image in images:
            imout.write(image)
            imout.write("\n")
        imout.close()
    else:
        fin = open("./namesLists/our_car_imageslist.txt", "r")
        images = fin.readlines()
        images = [x.replace("\n","") for x in images]
    for i, image in enumerate(images):
        for j in range(i, len(images)):
            fout = open("./verification_pairs/ourcar_verification_pairs_all.txt", "a")
            person1 = image.split("/")
            image2 = images[j]
            person2 = image2.split("/")
            if "caricature" in person1[0]:
                person1name = person1[0].replace("_caricature", "")
            else:
                person1name = person1[0]
            if "caricature" in person2[0]:
                person2name = person2[0].replace("_caricature", "")
            else:
                person2name = person2[0]
            if person1name == person2name:
                label = "1"
            else:
                label = "0"

            fout = open("./verification_pairs/ourcar_verification_pairs_all.txt", "a")
            fout.write(image)
            fout.write(" ")
            fout.write(image2)
            fout.write(" ")
            fout.write(label)
            fout.write("\n")
            fout.close()





#for testing
def webCaricature():
    images = []
    if os.path.exists("./namesLists/WebCaricature_imageslist.txt") == False:
        for root, dirs, files in os.walk("./WebCaricature/OriginalImages/"):
            if root != ("./WebCaricature/FacialPoints/"):
                #print(files)
                person = root.replace("./WebCaricature/OriginalImages/", "")
                for file in files:
                    image = person + "/" + file
                    images.append(image)
        images = sorted(images)
        imout = open("./namesLists/WebCaricature_imageslist.txt", "w")
        for image in images:
            imout.write(image)
            imout.write("\n")
        imout.close()
    else:
        fin = open("./namesLists/WebCaricature_imageslist.txt", "r")
        images = fin.readlines()
        images = [x.replace("\n", "") for x in images]
    numIms = len(images)
    ttlCountimage = 0
    for i, image in enumerate(images):
        for j in range(i, len(images)):
            fout = open("./verification_pairs/webcaricature_verification_pairs_all.txt", "a")
            person1 = image.split("/")
            image2 = images[j]
            person2 = image2.split("/")
            if person1[0] == person2[0]:
                label = "1"
            else:
                label = "0"
            fout.write(image)
            fout.write(" ")
            fout.write(image2)
            fout.write(" ")
            fout.write(label)
            fout.write("\n")
            fout.close()
#For train
def celebA():
    acceptableF = open("namesLists/acceptableCelebAIdentities_ourcar&webcar.txt", "r")
    acceptableIdentities = acceptableF.readlines()
    allimsID = open("namesLists/identity_CelebA.txt", "r")
    allIdentities = allimsID.readlines()

    allIdentities = sorted(allIdentities)

    goodID = []
    for identity in acceptableIdentities:
        identityName = identity.split()
        goodID.append(identityName[0])
    #fout = open("./verification_pairs/celebA_verification_pairs_all.txt", "a")
    images = []
    del acceptableIdentities
    for identity in allIdentities:
        imID = identity.split()
        if imID[1] in goodID:
            filePath = "id-" + imID[1] + "/" + imID[0]
            images.append(filePath)
    del goodID
    del allIdentities
    for i, image in enumerate(images):
        for j in range(i, len(images)):
            fout = open("./verification_pairs/celebA_verification_pairs_all.txt", "a")
            image2 = images[j]
            idIm = image.split("/")
            idIm2 = image2.split("/")
            if idIm[0] == idIm2[0]:
                label = "1"
            else:
                label = "0"

            fout.write(image)
            fout.write(" ")
            fout.write(image2)
            fout.write(" ")
            fout.write(label)
            fout.write("\n")
            fout.close()


if os.path.isdir("./verification_pairs/") == False:
    os.mkdir("./verification_pairs/")
ourCar()