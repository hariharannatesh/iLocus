#include<WiFi.h>
#include<string.h>
int r1,r2,r3,r4;
int count=0;
void setup()
{
  Serial.begin(115200);
  Serial1.begin(9600);
  WiFi.init();
  Serial.println("Welcome!");
  
  //listNetworks();
  
}
void loop()
{
  //delay(10);
  
 
    
    listNetworks();
 
  
  
}
void listNetworks()
{
  int numSsid = WiFi.scanNetworks();
   for (int thisNet=0; thisNet<numSsid;thisNet++)
   {
    if(strlen(WiFi.SSID(thisNet))==30)
    r1=WiFi.RSSI(thisNet);
    else if(strlen(WiFi.SSID(thisNet))==31)
    r2=WiFi.RSSI(thisNet);
    else if(strlen(WiFi.SSID(thisNet))==28)
    r3=WiFi.RSSI(thisNet);
    else if(strlen(WiFi.SSID(thisNet))==29)
    r4=WiFi.RSSI(thisNet);
   }
   Serial.print(r1);
   Serial.print(",");
     Serial.print(r2);
   Serial.print(",");
    Serial.print(r3);
    Serial.print(",");
    Serial.println(r4);
   
}

