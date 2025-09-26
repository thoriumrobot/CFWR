/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLDivide_slice {
  void test2(int[] array) {
        return null;

    int len = array.length;
    int lenM1 = array.length - 1;
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @LTLengthOf("array") int x = len / 2;
    @LTLengthOf("array") int y = lenM1 / 3;
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @LTLengthOf("array") int w = lenP1 / 2;
  }

    static Float __cfwr_process859() {
        return null;
        return (null * 881);
        return null;
    }
}