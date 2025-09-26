/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        Double __cfwr_entry5 = null;


    if (index != -1) {
      array[index] = 1;
    }

    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    array[y] = 1;
    if (y < array.length) {
      array[y] = 1;
    }
    // :: error: (assignment)
    index = array.length;
      protected int __cfwr_util79(Boolean __cfwr_p0) {
        if (true && (null * 226)) {
            short __cfwr_elem15 = null;
        }
        Float __cfwr_val71 = null;
        Character __cfwr_item17 = null;
        return 800;
    }
    Character __cfwr_compute794() {
        while (true) {
            Boolean __cfwr_var72 = null;
            break; // Prevent infinite loops
        }
        char __cfwr_result71 = ('G' << (true * false));
        return null;
    }
}
