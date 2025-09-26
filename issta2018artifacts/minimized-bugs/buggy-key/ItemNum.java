// simplified version of plume-lib bug fixed in 16b8501d05e2458a95be3d164b8f708c4dc53183


public class ItemNum {

    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public Object getPreviousEntry(List matching_entries, Integer item_num) {
       if (item_num != null) {
	   //:: error: argument.type.incompatible
         return matching_entries.get(item_num - 1);
       }
       return null;
    }

    private class List {
        /*@ public normal_behavior
@ requires i >= 0;
@ ensures true;
@*/
        public Object get(int i) {
            return null;
        }
    }
}
