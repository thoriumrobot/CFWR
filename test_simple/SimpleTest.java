import java.util.List;
import java.util.ArrayList;

public class SimpleTest {
    private int[] array;
    private int index;
    
    public SimpleTest(int[] arr, int idx) {
        array = arr;
        index = idx;
    }
    
    public void testMethod() {
        if (index < array.length) {
            array[index] = 42;
        }
    }
}
