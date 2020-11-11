import './App.css';
import React, {useState, useEffect, useCallback} from 'react';
import {useDropzone} from 'react-dropzone'

function App() {

  const [apiResponse, setApiResponse] = useState({});
  const [apiImgPath, setApiImgPath] = useState("");

  useEffect(() => {               //Why dont you put this in the post request below? For async reasons.
    if ("path" in apiResponse){
      setApiImgPath("http://localhost:9000/" + apiResponse?.path)
    }
  }, [apiResponse]);
            

  function sendFile(file){
    var formData = new FormData();
    var xhr      = new XMLHttpRequest();
    formData.append("profile_pic", file, file.name);
    //formData.append("btn_upload_profile_pic", "Upload")
    
    xhr.open("POST", "http://localhost:9000/uploads", true);
    xhr.setRequestHeader("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8");
    //xhr.setRequestHeader("Upgrade-Insecure-Requests", 1);
    xhr.onreadystatechange = function () {  
        if (xhr.readyState === 4) {  
            if (xhr.status === 200) {  
              setApiResponse(JSON.parse(xhr.responseText));
            } else {  
                console.error(xhr.statusText);
            }  
        }  
    };
    xhr.send(formData);
  }

  const onDrop = useCallback(acceptedFiles => {
    sendFile(acceptedFiles[0]);
  }, [])
  const {getRootProps, getInputProps, isDragActive} = useDropzone({onDrop})

  return (
    <div className="App">
      <div style={{marginLeft:"auto", marginRight:"auto", paddingLeft:"15px", paddingRight:"15px", maxWidth:"1140px"}}>
      {/* 
      <iframe title="dummyframe" name="dummyframe" id="dummyframe" style={{display: "none"}}></iframe>
      <form method="POST" action="http://localhost:9000/uploads" encType="multipart/form-data" target="dummyframe">
      <div>
        <label>Select your profile picture:</label>
        <input type="file" name="profile_pic" />
      </div>
        <div>
         <input type="submit" name="btn_upload_profile_pic" value="Upload" />
        </div>
      </form>
      */}

      <h1>Upload picture</h1>
      <section className="container" style={{marginBottom:"30px"}}>
        <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        {
          isDragActive ?
            <p>Drop the files here ...</p> :
            <p>Drag 'n' drop some files here, or click to select files</p>
        }
        </div>
      </section>

      
      <div style={{display:"flex", justifyContent:"space-around"}}>
        <img src={apiImgPath} alt="Upload in the field above" width="500" height="600" /> 
        <table style={{width:"600px"}}>
          <tbody>
          <tr>
            <th>PropertyIndex</th>
           <th>Value</th>
         </tr>
            {apiResponse?.values?.map((element,index) => {
               console.log(element)
               return (
                  <tr>
                    <td>{index}</td>
                    <td>{element}</td>
                 </tr>
                )
            })}
         </tbody>
        </table> 

      </div>
      </div>
    </div>
  );
}

export default App;
