# Define prompt
summary_prompt = """Write a summary of the following document using the format provided. The summary includes the title, publisher and published date if available, purpose and scope, key points, conclusion and implications, and key words. The summary should be comprehensive yet brief, aiming for a reading time of no more than one minute. Avoid any translation or substitution of actuarial terms in the document. When starting a summary, begin the summary with "Title:" without saying "Here is a summary". Here is the summary format:

Title: [title of the document]\n\nPublisher and Published Date: [published date and publisher's name]\n\nPurpose and Scope: [note the purpose and scope]\n\nKey Points:\n- [indicate key points in bullet point format]\n\nConclusions and Implications: [describe conclusion and implications for regulatory and practical purposes in the actuarial field]\n\nKey words: [indicate top five key words from the document]

Here is the document to summarize:

"{text}"
"""
