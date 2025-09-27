// Source-based slice around line 8
// Method: ArrayAssignmentSameLen#ArrayAssignmentSameLen(int[],int)

import org.checkerframework.checker.index.qual.*;

public class ArrayAssignmentSameLen {

  private final int[] i_array;
  private final @IndexFor("i_array") int i_index;

  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
    i_array = array;
    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
    int[] array = a;
    @LTLengthOf(
        value = {"array", "b"},
        offset = {"0", "-3"})
    // :: error: (assignment)
