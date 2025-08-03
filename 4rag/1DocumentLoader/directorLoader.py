from langchain_community.document_loaders import DirectoryLoader , TextLoader, PyMuPDFLoader

loader= DirectoryLoader(
    path="content",
    # glob="**/*.txt",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)

doc= loader.load()
# print(doc[0].page_content)
print(len(doc))
print(doc[0].metadata)
print(doc[500].page_content)  