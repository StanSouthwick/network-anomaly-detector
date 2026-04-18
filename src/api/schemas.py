from pydantic import BaseModel, Field

class FlowRecord(BaseModel):
    model_config = {"populate_by_name": True}
    Destination_Port: float
    Flow_Duration: float
    Total_Fwd_Packets: float
    Total_Length_of_Fwd_Packets: float
    Fwd_Packet_Length_Max: float
    Fwd_Packet_Length_Min: float
    Fwd_Packet_Length_Mean: float
    Bwd_Packet_Length_Max: float
    Bwd_Packet_Length_Min: float
    Flow_Bytes_s: float = Field(alias="Flow Bytes/s")
    Flow_Packets_s: float = Field(alias="Flow Packets/s")
    Flow_IAT_Mean: float
    Flow_IAT_Std: float
    Flow_IAT_Max: float
    Flow_IAT_Min: float
    Fwd_IAT_Mean: float
    Fwd_IAT_Std: float
    Fwd_IAT_Min: float
    Bwd_IAT_Total: float
    Bwd_IAT_Mean: float
    Bwd_IAT_Std: float
    Bwd_IAT_Max: float
    Bwd_IAT_Min: float
    Fwd_PSH_Flags: float
    Fwd_URG_Flags: float
    Fwd_Header_Length: float
    Bwd_Header_Length: float
    Fwd_Packets_s: float = Field(alias="Fwd Packets/s")
    Bwd_Packets_s: float = Field(alias="Bwd Packets/s")
    Min_Packet_Length: float
    Max_Packet_Length: float
    Packet_Length_Mean: float
    Packet_Length_Variance: float
    FIN_Flag_Count: float
    RST_Flag_Count: float
    PSH_Flag_Count: float
    ACK_Flag_Count: float
    URG_Flag_Count: float
    Down_Up_Ratio: float = Field(alias="Down/Up Ratio")
    Init_Win_bytes_forward: float
    Init_Win_bytes_backward: float
    act_data_pkt_fwd: float
    min_seg_size_forward: float
    Active_Mean: float
    Active_Std: float
    Active_Max: float
    Active_Min: float
    Idle_Std: float

class BatchRequest(BaseModel):
    records: list[FlowRecord]

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    anomaly_flagged: bool

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


