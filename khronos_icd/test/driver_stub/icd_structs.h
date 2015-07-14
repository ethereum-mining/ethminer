#ifndef _ICD_STRUCTS_H_
#define _ICD_STRUCTS_H_

typedef struct CLIicdDispatchTable_st  CLIicdDispatchTable;
typedef struct CLIplatform_st CLIplatform;

struct CLIicdDispatchTable_st
{
    void *entries[256];
    int entryCount;
};

struct CLIplatform_st
{
    CLIicdDispatchTable* dispatch;
};

#endif /* _ICD_STRUCTS_H_ */
