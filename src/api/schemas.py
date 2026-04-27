from pydantic import BaseModel, Field

class FlowRecord(BaseModel):
    model_config = {"populate_by_name": True}
    Destination_Port: float = Field(alias="Destination Port")
    Flow_Duration: float = Field(alias="Flow Duration")
    Total_Fwd_Packets: float = Field(alias="Total Fwd Packets")
    Total_Length_of_Fwd_Packets: float = Field(alias="Total Length of Fwd Packets")
    Fwd_Packet_Length_Max: float = Field(alias="Fwd Packet Length Max")
    Fwd_Packet_Length_Min: float = Field(alias="Fwd Packet Length Min")
    Fwd_Packet_Length_Mean: float = Field(alias="Fwd Packet Length Mean")
    Bwd_Packet_Length_Max: float = Field(alias="Bwd Packet Length Max")
    Bwd_Packet_Length_Min: float = Field(alias="Bwd Packet Length Min")
    Flow_Bytes_s: float = Field(alias="Flow Bytes/s")
    Flow_Packets_s: float = Field(alias="Flow Packets/s")
    Flow_IAT_Mean: float = Field(alias="Flow IAT Mean")
    Flow_IAT_Std: float = Field(alias="Flow IAT Std")
    Flow_IAT_Max: float = Field(alias="Flow IAT Max")
    Flow_IAT_Min: float = Field(alias="Flow IAT Min")
    Fwd_IAT_Mean: float = Field(alias="Fwd IAT Mean")
    Fwd_IAT_Std: float = Field(alias="Fwd IAT Std")
    Fwd_IAT_Min: float = Field(alias="Fwd IAT Min")
    Bwd_IAT_Total: float = Field(alias="Bwd IAT Total")
    Bwd_IAT_Mean: float = Field(alias="Bwd IAT Mean")
    Bwd_IAT_Std: float = Field(alias="Bwd IAT Std")
    Bwd_IAT_Max: float = Field(alias="Bwd IAT Max")
    Bwd_IAT_Min: float = Field(alias="Bwd IAT Min")
    Fwd_PSH_Flags: float = Field(alias="Fwd PSH Flags")
    Fwd_URG_Flags: float = Field(alias="Fwd URG Flags")
    Fwd_Header_Length: float = Field(alias="Fwd Header Length")
    Bwd_Header_Length: float = Field(alias="Bwd Header Length")
    Fwd_Packets_s: float = Field(alias="Fwd Packets/s")
    Bwd_Packets_s: float = Field(alias="Bwd Packets/s")
    Min_Packet_Length: float = Field(alias="Min Packet Length")
    Max_Packet_Length: float = Field(alias="Max Packet Length")
    Packet_Length_Mean: float = Field(alias="Packet Length Mean")
    Packet_Length_Variance: float = Field(alias="Packet Length Variance")
    FIN_Flag_Count: float = Field(alias="FIN Flag Count")
    RST_Flag_Count: float = Field(alias="RST Flag Count")
    PSH_Flag_Count: float = Field(alias="PSH Flag Count")
    ACK_Flag_Count: float = Field(alias="ACK Flag Count")
    URG_Flag_Count: float = Field(alias="URG Flag Count")
    Down_Up_Ratio: float = Field(alias="Down/Up Ratio")
    Init_Win_bytes_forward: float
    Init_Win_bytes_backward: float
    act_data_pkt_fwd: float
    min_seg_size_forward: float
    Active_Mean: float = Field(alias="Active Mean")
    Active_Std: float = Field(alias="Active Std")
    Active_Max: float = Field(alias="Active Max")
    Active_Min: float = Field(alias="Active Min")
    Idle_Std: float = Field(alias="Idle Std")

class BatchRequest(BaseModel):
    records: list[FlowRecord]

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    anomaly_flagged: bool

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


