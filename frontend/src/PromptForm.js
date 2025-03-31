import { useState } from "react";
import { useRef } from "react";
import { useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {PulseLoader} from "react-spinners";

function PromptForm(){
    const [prompt, setPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [messages, setMessages] = useState([]);
    // Use a ref to persist the websocket connection across renders.
    const websocketRef = useRef(null);


    // Create the websocket connection once when the component mounts.
    useEffect( () => {
        websocketRef.current = new WebSocket('ws://localhost:8000/async_chat');

        websocketRef.current.onopen = () => {
            console.log("WebSocket connected.");
        }

        // websocketRef.current.onmessage = (event) => {
        //     if (!event.data) return;
        //     let data = JSON.parse(event.data);
        //     if (!data || !data.role) return;
            
        //     if(data.role === 'model'){

        //         setIsLoading(false);
                
        //         setMessages(prev => {
        //             if (prev.length && prev[prev.length -1].role === 'model'){
        //                 const updatedLast = {
        //                     ...prev[prev.length - 1],
        //                     content: data.content
        //                 };
        //                 return [...prev.slice(0, prev.length -1), updatedLast];
        //             } else{
        //                 return [...prev, data];
        //             }
        //         }
        //         )
        //     }   
        //     else {
        //         if(data.role === 'video' || (typeof data.content === 'object' && data.content.error)){
        //             setIsLoading(false);
        //         }
        //         setMessages(prev => [...prev, data]);
        //     }

        // };


        websocketRef.current.onmessage = (event) => {
            if (!event.data) return;
            
            try {
                let data = JSON.parse(event.data);
                if (!data || !data.role) return;
                
                console.log("Received message:", data); // Add logging to debug
                
                if(data.role === 'video') {
                    // Video message received
                    console.log("Video message received:", data.content);
                    setIsLoading(false);
                    setMessages(prev => [...prev, data]);
                }
                else if(data.role === 'model') {
                    setIsLoading(false);
                    
                    setMessages(prev => {
                        if (prev.length && prev[prev.length -1].role === 'model'){
                            const updatedLast = {
                                ...prev[prev.length - 1],
                                content: data.content
                            };
                            return [...prev.slice(0, prev.length -1), updatedLast];
                        } else {
                            return [...prev, data];
                        }
                    });
                }
                else {
                    setMessages(prev => [...prev, data]);
                }
            } catch (error) {
                console.error("Error processing message:", error);
            }
        };





        websocketRef.current.onclose = (event) => {
            console.log("WebSocket disconnected.");
        };

        // Clean up the connection when the component unmounts.
        return () => {
            if (websocketRef.current){
                websocketRef.current.close();
            }
        };
    }, []);

    const handleSubmit = async (e) => {
        setIsLoading(true);
        e.preventDefault();

        // setMessages(prev => [...prev, {role: 'user', content: prompt}]);

        if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN){
            websocketRef.current.send(prompt);
        } else {
            console.error("WebSocket not connected.");
        }
        setPrompt('');
    }

    return(
        <div className="main-container">
            <div id="conversation">
                {messages.map((message, index) => (
                    <div key={index} className={`message ${message.role}`}>
                        {
                        message.role === 'video' ? (
                            <>  
                                <p><b>Your Video</b></p>
                                <video 
                                    controls 
                                    width="50%" 
                                    preload="auto"
                                    src={message.content}
                                    onError={(e) => {
                                        console.error("Video loading error:", e);
                                        // Try to reload video on error
                                        const vid = e.target;
                                        vid.load();
                                    }}
                                    onLoadedData={() => console.log("Video loaded successfully")}
                                ></video>
                            </>
                        ) :
                        
                        // message.role === 'video'? (
                        //     <>  
                        //         <p><b>Your Video</b></p>
                        //         <video controls width="50%" src={message.content}></video>
                        //     </>
                        // ) : 
                        // typeof message.content === 'object' && message.content.video_url? (
                        //     <>
                        //     </>
                        // ) : 
                        typeof message.content === 'object' && message.content.error ? (
                            <>
                                {console.log(message.content.error)}
                            </>
                        ) :
                        message.role === 'user' && JSON.stringify(message.content).includes("Title")? (  //to prevent showing exra information sent by the search tool
                            <>  
                            </>
                        ) :
                        message.role === 'user'? (
                            <>
                                <p><b>You asked</b></p>
                                <p>{message.content}</p>
                            </>
                        ) :
                        (
                        <>
                            <p><b>AI Response</b></p>
                            <ReactMarkdown>{message.content}</ReactMarkdown>
                        </>
                        )
                        }
                    </div>
                ))}
            </div>
            {isLoading && (
                    <div className="loader-container">
                        <PulseLoader color="#3498db" />
                    </div>
                )}
            <form className="form">
                <input className="form-input" type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)}/>
                <button  className="form-button"type="submit" onClick={handleSubmit}>Submit</button>
            </form>
        </div>
    )
}

export default PromptForm;