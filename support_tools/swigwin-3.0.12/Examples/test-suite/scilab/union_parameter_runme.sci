// Some lines are commented out because of too long identifiers...

exec("swigtest.start", -1);

event = new_SDL_Event();

for i=1:2
  evAvailable = SDL_PollEvent(event);
  evType = SDL_Event_type_get(event);
  
  if evType==1 then
    specEvent = SDL_Event_active_get(event);
    _type = SDL_ActiveEvent_type_get(specEvent);
    
    if _type <> evType then swigtesterror(); end
    
    gain = SDL_ActiveEvent_gain_get(specEvent);
    //state = SDL_ActiveEvent_state_get(specEvent);
  end
  
  if evType==2 then
    specEvent = SDL_Event_key_get(event);
    //_type = SDL_KeyboardEvent_type_get(specEvent);
    
    //if _type <> evType then swigtesterror(); end
    
    //_which = SDL_KeyboardEvent_which_get(specEvent);
    //state = SDL_KeyboardEvent_state_get(specEvent);
  end
  
end

delete_SDL_Event(event);

exec("swigtest.quit", -1);