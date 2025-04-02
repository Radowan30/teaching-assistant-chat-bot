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
<>
      {/* Fixed Conversation Area: starts right below the header and above the prompt input */}
      <div 
        id="conversation" 
        className="fixed top-20 bottom-16 left-0 right-0 max-w-4xl mx-auto px-4 overflow-y-auto pt-4"
      >
        {messages.map((message, index) => {
          // Align user messages to the right and others to the left
          const alignment = message.role === 'user' ? 'items-end' : 'items-start';
          // Skipping rendering messages that include "Title" in user messages to not show extra information provided by the Web Search tool
          if(message.role === 'user' && JSON.stringify(message.content).includes("Title")){
            return null;
          }
          return (
            <div key={index} className={`flex flex-col ${alignment} mb-4`}>
              <span className="mb-1 text-s font-bold text-gray-700">
                {message.role === 'video' ? "Your Video" : message.role === 'user' ? "You Asked" : "AI Response"}
              </span>
              <div className="max-w-md bg-white p-4 rounded-lg shadow-md text-left">
                {message.role === 'video' ? (
                  <video
                    controls
                    className="w-full mx-auto"
                    preload="auto"
                    src={message.content}
                    onError={(e) => {
                      console.error("Video loading error:", e);
                      const vid = e.target;
                      vid.load();
                    }}
                    onLoadedData={() => console.log("Video loaded successfully")}
                  ></video>
                ) : message.role === 'user' ? (
                  <p>{message.content}</p>
                ) : (
                  <ReactMarkdown
                    components={{
                      a: ({node, ...props}) => (
                        <a className="text-blue-400" {...props} />
                      )
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                )}
              </div>
            </div>
          );
        })}
        {isLoading && (
          <div className="flex items-start mb-4">
            <div className="p-4">
              <PulseLoader color="#3498db" />
            </div>
          </div>
        )}
      </div>
      {/* Fixed Prompt Input at the bottom */}
      <form 
        className="fixed bottom-0 left-0 right-0 max-w-4xl mx-auto bg-white border-t shadow-md p-4 flex items-center gap-2"
        onSubmit={handleSubmit}
      >
        <input
          className="flex-1 border border-gray-400 rounded p-2 shadow-md"
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Type your prompt"
        />
        <button 
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition"
          type="submit"
        >
          Send
        </button>
      </form>
    </>
    )
}

export default PromptForm;